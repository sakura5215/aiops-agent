"""自研 Agent Loop 治理逻辑单测

  [MOCK] 步数上限
  [MOCK] 重复调用检测
  [MOCK] 上下文压缩
  [MOCK] token 阈值触发压缩
  [MOCK] 报告信号切换（fill_context_for_report 触发 report_mode）
  [MOCK] 幻觉工具 / 未知工具兜底

通过注入 fake llm（mock 掉 bind_tools 后的 invoke）和 mock 摘要器来驱动循环分支。

运行：.venv/Scripts/python.exe tests/test_react_loop.py
"""
import sys
import os
import json

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage

import agent.react_loop as rl
from agent.react_loop import ReActLoop, _execute_tool, _compress_observations

_passed = 0
_failed = 0


def check(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  [PASS] {name}")
    else:
        _failed += 1
        print(f"  [FAIL] {name}  {detail}")


def _ai_with_tools(name, args, tool_id="t1"):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": tool_id, "type": "tool_call"}],
    )


def _ai_final(text="完成"):
    return AIMessage(content=text)


class _FakeLLM:
    """按脚本依次吐出响应，用完回落到最终回答。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.script:
            return self.script.pop(0)
        return _ai_final()


def _make_loop(llm_script, **kw):
    loop = ReActLoop()
    loop.llm = _FakeLLM(llm_script)
    loop.max_steps = kw.get("max_steps", 8)
    loop.compress_after_steps = kw.get("compress_after_steps", 6)
    loop.compress_token_threshold = kw.get("compress_token_threshold", 4000)
    loop.dedup_max_repeat = kw.get("dedup_max_repeat", 2)
    # 默认注入 mock 摘要器（离线可跑），返回带标记的摘要便于断言
    loop.summarizer = kw.get("summarizer", lambda text: "【摘要】" + text[:20])
    # 默认注入确定性 token 计数器（离线可跑，不依赖 Qwen3 tokenizer 下载）
    loop.token_counter = kw.get("token_counter", lambda text: max(1, len(text) // 2))
    return loop


def test_max_steps():
    # 模型永远返回 tool_call，永不终止 → 应被步数上限强制掐断
    loop = _make_loop([_ai_with_tools("get_target_service", {})] * 10, max_steps=3)
    text, trace = loop.run([SystemMessage(content="s"), HumanMessage(content="q")])
    check("max_steps 强制终止", trace.terminal_reason == "max_steps" and not trace.finished, trace.summary())
    check("max_steps 返回治理提示", "最大推理步数" in text, text[:40])


def test_dedup_repeat():
    # 同一工具+参数连续调 2 次 → 第二次的 ToolMessage 应带重复提示
    loop = _make_loop([
        _ai_with_tools("get_target_service", {}),
        _ai_with_tools("get_target_service", {}),
        _ai_final("已换思路完成"),
    ], dedup_max_repeat=2)
    text, trace = loop.run([SystemMessage(content="s"), HumanMessage(content="q")])
    # 从最后一次模型调用拿到的 messages 里找 ToolMessage，检查是否含重复提示
    last_call = loop.llm.calls[-1]
    hint_found = any(
        isinstance(m, ToolMessage) and "重复" in (m.content or "")
        for m in last_call
    )
    check("重复调用注入提示", hint_found, f"tool 消息数={sum(1 for m in last_call if isinstance(m, ToolMessage))}")


def test_compress():
    # 5 条 tool 消息 → 压缩后首条为 LLM 摘要、其余占位，且 tool_call_id 保留
    msgs = [
        ToolMessage(content="r1", tool_call_id="c1"),
        ToolMessage(content="r2", tool_call_id="c2"),
        ToolMessage(content="r3", tool_call_id="c3"),
    ]
    calls = []
    _compress_observations(msgs, 0, 3, summarizer=lambda t: calls.append(t) or "【LLM摘要】")
    check("压缩后首条为 LLM 摘要", msgs[0].content == "【LLM摘要】", msgs[0].content[:30])
    check("压缩后其余占位", "已并入上文摘要" in msgs[1].content and "已并入上文摘要" in msgs[2].content)
    check("tool_call_id 保留", all(m.tool_call_id for m in msgs))
    check("摘要器收到全部工具结果", len(calls) == 1 and "r1" in calls[0] and "r3" in calls[0], repr(calls[0][:40]) if calls else "未调用")


def test_token_trigger_compress():
    # 用 token 阈值触发：前两轮积累 tool 消息，第三轮时 token 累计超阈值触发压缩
    summary_log = []
    loop = _make_loop(
        [
            _ai_with_tools("fetch_metric_data", {"service_name": "order-service", "time_range": "最近1小时"}),
            _ai_with_tools("fetch_log_summary", {"service_name": "order-service", "time_range": "最近1小时"}),
            _ai_with_tools("fetch_alert_data", {"service_name": "order-service", "time_range": "最近1小时"}),
            _ai_final("完成"),
        ],
        compress_after_steps=99,  # 把步数阈值调高，确保是 token 阈值触发
        compress_token_threshold=150,  # 累积几轮 tool 结果后超阈值
        summarizer=lambda t: summary_log.append(t) or "【摘要】",
    )
    text, trace = loop.run([SystemMessage(content="s"), HumanMessage(content="q")])
    check("token 阈值触发压缩", trace.compressed, trace.summary())
    check("摘要器被调用", len(summary_log) >= 1, f"调用次数={len(summary_log)}")


def test_report_signal():
    # 调用 fill_context_for_report 后，下一轮应切报告提示词
    from utils.prompt_loader import load_report_prompts, load_system_prompts
    loop = _make_loop([
        _ai_with_tools("fill_context_for_report", {}),
        _ai_final("报告生成完毕"),
    ])
    text, trace = loop.run([SystemMessage(content=load_system_prompts()), HumanMessage(content="q")])
    # 找最后一次模型调用时 messages 里首条 system 是否为报告提示词
    last_call = loop.llm.calls[-1]
    first_sys = last_call[0].content
    report_prompt = load_report_prompts()
    check("报告信号触发提示词切换", first_sys == report_prompt, first_sys[:30])


def test_unknown_tool():
    # 幻觉工具 → 返回兜底错误，不抛异常
    r = _execute_tool("not_exist_tool", {})
    check("幻觉工具兜底", r.startswith("ERROR: 未知工具"), r[:30])


def test_real_tool_execute():
    # 真实工具可执行（fetch_metric_data 走 mock 数据）
    r = _execute_tool("fetch_metric_data", {"service_name": "order-service", "time_range": "最近1小时"})
    check("真实工具执行", "cpu_usage" in r and "83%" in r, r[:50])


if __name__ == "__main__":
    test_max_steps()
    test_dedup_repeat()
    test_compress()
    test_token_trigger_compress()
    test_report_signal()
    test_unknown_tool()
    test_real_tool_execute()
    print(f"\n总计: {_passed} passed, {_failed} failed")
    sys.exit(1 if _failed else 0)
