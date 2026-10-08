"""自写 ReAct 循环。create_agent 把循环包在框架里，很多细节看不到也改不了，
所以这里直接自己写一遍循环，顺带把几个实际问题处理掉：

1. 步数上限（max_steps）      —— 防止模型在某个工具上反复打转，超了就强制结束。
2. 重复调用检测（dedup）      —— 同一个工具 + 同样的参数连着调，就提示模型换个思路。
3. 上下文压缩（compress）     —— 步数或 token 超了阈值，把前面的工具结果用 LLM
                                 摘要成一段，别让上下文越滚越大。

循环本身：
    messages ──调模型──▶ 有没有 tool_calls? ──有──▶ 执行工具 ──把结果塞回 messages ──▶ 回到开头
                                            └─没有──▶ 直接返回最终回答

跟 create_agent 版共用同一套 9 个工具和双场景提示词（fill_context_for_report
触发报告提示词切换），config/agent.yml 里 custom_loop_enabled 可以切回原版对照。

token 计数用 utils/token_counter.py 里的 Qwen3 tokenizer（跟 qwen3-max 同一套
词表），不是 GPT-2 那种粗略估的；只在没网/没装依赖时才退回 GPT-2 并记个日志。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage

from agent.tools.agent_tools import (
    rag_summarize,
    get_target_service,
    get_time_range,
    fetch_alert_data,
    fetch_metric_data,
    fetch_log_summary,
    fetch_service_topology,
    fetch_report_data,
    fill_context_for_report,
)
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts, load_report_prompts
from utils.config_handler import agent_conf
from utils.logger_handler import logger
from utils.token_counter import count_tokens

# 工具注册：名字 -> 可调用对象。langchain @tool 装饰后的对象本身可调用（含 schema）。
TOOL_FUNCTIONS: dict[str, Any] = {
    "rag_summarize": rag_summarize,
    "get_target_service": get_target_service,
    "get_time_range": get_time_range,
    "fetch_alert_data": fetch_alert_data,
    "fetch_metric_data": fetch_metric_data,
    "fetch_log_summary": fetch_log_summary,
    "fetch_service_topology": fetch_service_topology,
    "fetch_report_data": fetch_report_data,
    "fill_context_for_report": fill_context_for_report,
}

# 信号工具：调用它意味着「本次任务是要生成报告」，触发报告场景提示词切换。
REPORT_SIGNAL_TOOL = "fill_context_for_report"


@dataclass
class LoopTrace:
    """记录一次任务跑的过程，方便之后排查（每一步调了哪个工具、怎么结束的）。"""

    steps: list[dict] = field(default_factory=list)
    max_steps: int = 0
    finished: bool = False
    compressed: bool = False
    terminal_reason: str = ""  # "final_answer" | "max_steps"

    def record(self, entry: dict) -> None:
        self.steps.append(entry)

    def summary(self) -> str:
        tool_names = [s.get("tool") for s in self.steps if s.get("tool")]
        return (
            f"steps={len(self.steps)} finished={self.finished} "
            f"compressed={self.compressed} reason={self.terminal_reason} "
            f"tools={tool_names}"
        )


def _execute_tool(name: str, args: dict) -> str:
    """执行单个工具调用，返回字符串结果（未知工具 / 异常时返回错误文案）。"""
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        # 模型编了个不存在的工具名
        return f"ERROR: 未知工具 {name}，请从已提供的工具中选择"
    try:
        result = fn.invoke(args) if hasattr(fn, "invoke") else fn(**args)
        return str(result)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[react_loop]工具 {name} 执行异常: {e}")
        return f"ERROR: 工具 {name} 执行失败：{e}"


def _summarize_tool_results(tool_texts: list[str], summarizer, retries: int = 1) -> str:
    """把一批工具结果丢给 LLM 提炼成一段摘要。

    summarizer 是「接收文本、返回摘要」的可调用对象，默认用项目 chat_model，
    测试时换成 mock。LLM 偶发超时或限流会先重试一次，还不行就退回原文拼接，
    保证压缩失败也不丢信息。
    """
    joined = "\n".join(f"- {t}" for t in tool_texts)
    prompt = (
        "请把以下多步工具查询/检索的结果提炼成要点摘要，保留关键数据、异常结论与"
        "排查线索，去掉重复和无关细节，控制在 300 字以内：\n" + joined
    )
    last_err = None
    for attempt in range(retries + 1):
        try:
            return summarizer(prompt)
        except Exception as e:  # noqa: BLE001
            last_err = e
            logger.warning(
                f"[react_loop]摘要失败（第 {attempt + 1}/{retries + 1} 次）: {e}"
            )
    logger.warning(f"[react_loop]摘要重试仍失败，降级为原文拼接: {last_err}")
    return "[已压缩的历史工具结果] 要点汇总：\n" + joined


def _compress_observations(
    messages: list[BaseMessage], start: int, end: int, summarizer=None
) -> None:
    """把 [start, end) 区间的 tool 消息摘要成一条，替换回原位。

    只动 role=tool 的消息；assistant 里的 tool_calls 结构（含 tool_call_id）要
    留着，不然消息对不上。做法：把这个区间所有 tool 内容交给 LLM 提炼成一段，
    写进第一条 tool 消息，其余 tool 消息内容清成占位。这样既省 token，结论也没丢。
    """
    tool_texts = []
    for i in range(start, end):
        m = messages[i]
        if isinstance(m, ToolMessage):
            tool_texts.append(m.content if isinstance(m.content, str) else str(m.content))
    if not tool_texts:
        return
    if summarizer is None:
        summarizer = _default_summarizer()
    summary = _summarize_tool_results(tool_texts, summarizer)
    first_tool_written = False
    for i in range(start, end):
        m = messages[i]
        if isinstance(m, ToolMessage):
            if not first_tool_written:
                messages[i] = ToolMessage(
                    content=summary, tool_call_id=m.tool_call_id
                )
                first_tool_written = True
            else:
                messages[i] = ToolMessage(
                    content="(内容已并入上文摘要)", tool_call_id=m.tool_call_id
                )


def _default_summarizer():
    """返回一个用项目 chat_model 做摘要的可调用对象（延迟 import，避免循环依赖）。"""
    def summarize(text: str) -> str:
        from langchain_core.messages import HumanMessage
        from model.factory import chat_model as cm
        resp = cm.invoke([HumanMessage(content=text)])
        content = resp.content if hasattr(resp, "content") else str(resp)
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("text"):
                    parts.append(block["text"])
                elif isinstance(block, str):
                    parts.append(block)
            content = "".join(parts)
        return str(content)
    return summarize


class ReActLoop:
    """自己写的 ReAct 循环，只用 LangChain 的消息类型和模型绑定，不套上层框架。"""

    def __init__(self):
        # 绑定 9 个工具，tool_choice=auto：模型自由决定调不调工具
        self.llm = chat_model.bind_tools(list(TOOL_FUNCTIONS.values()), tool_choice="auto")
        self.max_steps = int(agent_conf.get("loop_max_steps", 8))
        self.compress_after_steps = int(agent_conf.get("compress_after_steps", 6))
        # token 阈值：上下文累计超过该值即触发压缩（与步数阈值「任一命中」即压）
        self.compress_token_threshold = int(agent_conf.get("compress_token_threshold", 4000))
        self.dedup_max_repeat = int(agent_conf.get("dedup_max_repeat", 2))
        # 摘要器：默认走 chat_model，测试可注入 mock
        self.summarizer = _default_summarizer()
        # token 计数器：默认 Qwen3 精确计数，测试可注入确定性 mock（离线可跑）
        self.token_counter = count_tokens

    def run(self, messages: list[BaseMessage]) -> tuple[str, LoopTrace]:
        """执行一次任务。输入 messages 已含 system + 历史 + 用户提问（调用方负责拼接）。

        返回 (final_text, trace)。
        """
        trace = LoopTrace(max_steps=self.max_steps)
        # 报告场景状态：一旦信号工具被调用，后续轮次切报告提示词
        report_mode = False
        # 重复调用检测：key = (tool_name, args_json)，value = 连续出现次数
        last_calls: dict[str, int] = {}

        # messages 可能以 SystemMessage 开头；循环中我们只在顶部维护 system 与内容。
        # 为简化，这里假设 messages 的 system 已由调用方注入（含长期记忆召回），
        # 我们只在切报告提示词时替换掉首条 system。
        for step in range(1, self.max_steps + 1):
            # 根据当前是否报告模式，替换 system 提示词
            self._apply_prompt_mode(messages, report_mode)

            resp = self.llm.invoke(messages)
            ai_msg = resp if isinstance(resp, AIMessage) else AIMessage(content=str(resp))
            messages.append(ai_msg)

            tool_calls = getattr(ai_msg, "tool_calls", None)
            if not tool_calls:
                trace.finished = True
                trace.terminal_reason = "final_answer"
                return self._extract_text(ai_msg), trace

            # 执行本轮所有工具调用
            for tc in tool_calls:
                name = tc.get("name", "")
                raw_args = tc.get("args", {}) or {}
                args = raw_args if isinstance(raw_args, dict) else json.loads(str(raw_args))

                # 信号工具：置报告模式
                if name == REPORT_SIGNAL_TOOL:
                    report_mode = True

                # 重复调用检测
                dedup_key = f"{name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
                last_calls[dedup_key] = last_calls.get(dedup_key, 0) + 1
                repeat_hint = ""
                if last_calls[dedup_key] >= self.dedup_max_repeat:
                    repeat_hint = (
                        "（注意：该工具与参数已连续调用多次，若仍无新信息请换思路，"
                        "不要重复同一调用）"
                    )

                result = _execute_tool(name, args)
                messages.append(
                    ToolMessage(content=result + repeat_hint, tool_call_id=tc.get("id", ""))
                )
                trace.record({"step": step, "tool": name, "args": args})

            # 步数或 token 超了阈值就压缩一次
            if not trace.compressed and self._should_compress(step, messages):
                if self._compress_if_needed(messages):
                    trace.compressed = True

        trace.terminal_reason = "max_steps"
        return (
            "已达到最大推理步数，任务可能未完成（步数上限治理已触发）。"
            "请精简提问或拆分任务后重试。",
            trace,
        )

    def _apply_prompt_mode(self, messages: list[BaseMessage], report_mode: bool) -> None:
        """切换 system 提示词：报告模式用报告提示词，否则用运维问答提示词。

        长期记忆是通过一条额外的 SystemMessage 注入的，切换提示词时不能把它顶掉，
        所以这里保留最后一条 system（长期记忆），只替换最前面的场景提示词。
        """
        prompt = load_report_prompts() if report_mode else load_system_prompts()
        # 找到所有 system 消息的索引
        sys_idxs = [i for i, m in enumerate(messages) if isinstance(m, SystemMessage)]
        if not sys_idxs:
            messages.insert(0, SystemMessage(content=prompt))
            return
        # 保留最后一条 system（长期记忆），其余替换为场景提示词
        keep = messages[sys_idxs[-1]]
        # 场景提示词放最前，长期记忆紧跟其后
        new_sys = SystemMessage(content=prompt)
        # 删除所有旧 system
        for i in reversed(sys_idxs):
            del messages[i]
        messages.insert(0, new_sys)
        if keep is not None:
            messages.insert(1, keep)

    def _should_compress(self, step: int, messages: list[BaseMessage]) -> bool:
        """判断要不要压缩：步数到了，或者非 system 消息的 token 累计超了阈值。

        token 用 Qwen3 tokenizer 精确数（utils/token_counter.py），没网/没装依赖
        时才退回 GPT-2 估。只数非 system 消息，避免固定提示词把阈值撑爆。
        """
        if step >= self.compress_after_steps:
            return True
        try:
            total = 0
            for m in messages:
                if isinstance(m, SystemMessage):
                    continue
                content = m.content if isinstance(m.content, str) else str(m.content)
                total += self.token_counter(content)
            return total >= self.compress_token_threshold
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[react_loop]token 计数异常，退化为步数触发: {e}")
            return False

    def _compress_if_needed(self, messages: list[BaseMessage]) -> bool:
        """真压缩：把第一条到最后一条之间的 tool 消息合并，留最后一条。

        返回是否真的压了（tool 消息太少就直接不压，返回 False）。
        """
        tool_idxs = [i for i, m in enumerate(messages) if isinstance(m, ToolMessage)]
        if len(tool_idxs) < 3:  # 少于 3 条 tool 没必要压
            return False
        # 压缩 [first, last) 之间的 tool 消息，留最后一条
        start, end = tool_idxs[0], tool_idxs[-1]
        if end - start < 2:
            return False
        logger.info(f"[react_loop]触发上下文压缩：合并 {len(tool_idxs)} 条 tool 消息")
        _compress_observations(messages, start, end, summarizer=self.summarizer)
        return True

    @staticmethod
    def _extract_text(ai_msg: AIMessage) -> str:
        content = ai_msg.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("text"):
                    parts.append(block["text"])
                elif isinstance(block, str):
                    parts.append(block)
            return "".join(parts)
        return str(content)
