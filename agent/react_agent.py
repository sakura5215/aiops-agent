import time

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from agent.memory import FileChatMessageHistory

from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from utils.config_handler import agent_conf
from utils.logger_handler import logger
from agent.tools.agent_tools import *
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch


class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[
                rag_summarize,
                get_target_service,
                get_time_range,
                fetch_alert_data,
                fetch_metric_data,
                fetch_log_summary,
                fetch_service_topology,
                fetch_report_data,
                fill_context_for_report,
            ],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )
        # 长期记忆：优先 viking 分层记忆（mem0 治理 + viking 分层组织与检索），
        # 可用 config/agent.yml 的 viking_enabled 关掉回退到 mem0 扁平版，
        # 两者都初始化失败时降级关闭，不阻断 agent 启动
        self.viking = None
        self.mem0_store = None
        if agent_conf.get("long_term_memory_enabled", True):
            use_viking = agent_conf.get("viking_enabled", True)
            try:
                if use_viking:
                    from agent.viking import VikingMemoryStore
                    self.viking = VikingMemoryStore()
                    logger.info("[ReactAgent]viking 分层记忆已启用")
                else:
                    from agent.memory_store import MemoryStore
                    self.mem0_store = MemoryStore()
                    logger.info("[ReactAgent]mem0 扁平记忆已启用")
            except Exception as e:
                logger.warning(f"[ReactAgent]viking 初始化失败，回退 mem0 扁平记忆: {e}")
                self.viking = None
                # 降级要落到能用的那一条：viking 起不来不等于 mem0 也起不来
                try:
                    from agent.memory_store import MemoryStore
                    self.mem0_store = MemoryStore()
                    logger.info("[ReactAgent]已回退到 mem0 扁平记忆")
                except Exception as e2:
                    logger.warning(f"[ReactAgent]mem0 兜底也失败，长期记忆降级关闭: {e2}")

    def _get_session_history(self, session_id: str) -> FileChatMessageHistory:
        # 每个对话id对应一个不同文件
        return FileChatMessageHistory(session_id=session_id, storage_dir="./chat_histories")

    def _get_ai_message(self, result) -> AIMessage | None:
        """从结果中提取最后一个AIMessage，用于存入历史"""
        if isinstance(result, dict):
            messages = result.get("messages",[])
            if messages:
                last = messages[-1]
                if isinstance(last, AIMessage):
                    return last
                # 如果最后不是AIMessage（例如是FunctionMessage），可以遍历栈
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage):
                        return msg
        return None

    def _extract_text(self, result) -> str:
        if isinstance(result, dict):
            messages = result.get("messages", [])
            if messages:
                last = messages[-1]
                content = getattr(last, "content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    texts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("text"):
                            texts.append(block["text"])
                    return "".join(texts)
        return str(result)

    def execute_stream(self, session_id, query):
        history = self._get_session_history(session_id)

        # 将用户的问题加入历史文件
        user_msg = HumanMessage(content=query)
        history.add_message(user_msg)

        # 滑动窗口短期记忆：仅取最近 K 条历史喂回，避免长对话 token 膨胀
        full_messages = history.recent_messages(agent_conf.get("memory_window", 20))

        # 长期记忆检索：按当前 query 召回相关事实，作为上下文注入
        # 注意：这里必须走 viking / mem0 两条真实分支。曾经写成 self.memory_store，
        # 该属性在切 viking 时已被删掉，异常被 except 吞掉只留一条 warning，
        # 结果长期记忆"一直在跑但永远召回为空"——静默失败，比直接报错更危险
        store = self.viking or self.mem0_store
        if agent_conf.get("long_term_memory_enabled", True) and store:
            try:
                if self.viking:
                    recalled = self.viking.recall(query, k=agent_conf.get("memory_recall_k", 3))
                else:
                    recalled = self.mem0_store.search(
                        query, k=agent_conf.get("memory_recall_k", 3)
                    )
                if recalled:
                    memory_context = "以下是与本次提问相关的长期记忆事实，供参考：\n" + \
                        "\n".join(f"- {f}" for f in recalled)
                    full_messages = [SystemMessage(content=memory_context)] + full_messages
            except Exception as e:
                logger.warning(f"[ReactAgent]长期记忆检索失败，跳过: {e}")

        # 非流式调用，先拿完整结果
        result = self.agent.invoke(
            {"messages": full_messages},
            context={"report": False},
        )

        # 提取并保存助手回复到历史文件
        ai_message = self._get_ai_message(result)
        if ai_message:
            history.add_message(ai_message)     # 会自动序列化写入文件
            # 长期记忆写入：从本轮交流抽取原子事实去重入库
            if agent_conf.get("long_term_memory_enabled", True) and (self.viking or self.mem0_store):
                try:
                    if self.viking:
                        self.viking.commit([user_msg, ai_message], session_id=str(session_id))
                    else:
                        self.mem0_store.add([user_msg, ai_message], user_id=str(session_id))
                except Exception as e:
                    logger.warning(f"[ReactAgent]长期记忆写入失败，跳过: {e}")

        final_text = self._extract_text(result)

        # 伪流式：按固定长度切块输出
        chunk_size = 20
        for i in range(0, len(final_text), chunk_size):
            yield final_text[i:i + chunk_size]
            time.sleep(0.02)


if __name__ == '__main__':
    agent = ReactAgent()
    for chunk in agent.execute_stream(1, "分析一下 order-service 最近1小时的异常情况"):
        print(chunk, end="", flush=True)
