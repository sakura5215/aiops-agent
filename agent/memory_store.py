"""
mem0 式长期记忆：轻量实现两阶段流水线，借鉴 mem0 的设计但不引入其完整依赖。

- 提取阶段：每轮对话后用 LLM 从最新交流抽取"值得长期记住的原子事实"
- 更新阶段：新事实向量化检索最相似的已有事实，由 LLM 决定 ADD / UPDATE / MERGE / DELETE
- 检索阶段：下一轮 query 向量化召回 top-K 事实，作为长期记忆注入 prompt

与 mem0 一致采用 ADD-only 倾向：默认追加，矛盾事实并存，靠检索时的元数据时间排序
让新事实在 prompt 中优先呈现。事实原文与元数据以 JSON 文件为 source of truth，
Milvus 向量库作为检索索引，更新时全量重建（v1 简化实现，增量重建见 README Roadmap）。
"""
import os
import json
import time

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from model.factory import embed_model, chat_model
from utils.path_tool import get_abs_path
from utils.logger_handler import logger

MEMORY_COLLECTION = "agent_memory"
MEMORY_JSON = "memory_store/memory.json"

FACT_EXTRACTION_PROMPT = """你是一个记忆事实抽取器。从下面的对话片段中抽取"值得长期记住的原子事实"，每条一行。
只抽取：用户偏好、决策、长期有效的结论、关键运维结论（如某服务某时段的根因定位）。
不要抽取：临时计算、寒暄、一次性查询、工具中间结果。
若没有值得记住的事实，只输出：无
对话片段：
{dialog}
事实（每行一条，无编号）："""

UPDATE_PROMPT = """判断新事实与已有事实的关系，从以下操作中选一个，只输出操作名（大写）：
- ADD：新事实与已有事实不同，是新信息
- UPDATE：新事实是对已有事实所描述对象的更新（同一对象的新状态）
- MERGE：新事实可与已有事实合并成更完整的一条
- DELETE：新事实是已有事实的重复或冗余
新事实：{new_fact}
已有事实：{old_fact}
操作："""


class MemoryStore:
    """长期记忆库：事实抽取 + 去重合并 + 相似度检索。"""

    def __init__(self):
        self.uri = get_abs_path("milvus_data/milvus.db")
        os.makedirs(os.path.dirname(self.uri), exist_ok=True)
        self.collection = MEMORY_COLLECTION
        self.json_path = get_abs_path(MEMORY_JSON)
        os.makedirs(os.path.dirname(self.json_path), exist_ok=True)
        if not os.path.exists(self.json_path):
            self._save_facts([])
        self.model = chat_model
        self.vector_store = self._build_vector_store()

    def _build_vector_store(self):
        """构建 Milvus 向量库连接（独立 collection，与知识库 RAG 的 collection 隔离）。"""
        from langchain_milvus import Milvus
        return Milvus(
            embedding_function=embed_model,
            collection_name=self.collection,
            connection_args={"uri": self.uri},
        )

    def _load_facts(self) -> list[dict]:
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_facts(self, facts: list[dict]):
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(facts, f, ensure_ascii=False, indent=2)

    def _extract_facts(self, messages: list[BaseMessage]) -> list[str]:
        """提取阶段：用 LLM 从对话片段抽取原子事实。"""
        dialog = "\n".join(
            f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}"
            for m in messages
            if isinstance(m, (HumanMessage, AIMessage)) and getattr(m, "content", "")
        )
        if not dialog.strip():
            return []
        try:
            resp = self.model.invoke(FACT_EXTRACTION_PROMPT.format(dialog=dialog))
            content = resp.content if isinstance(resp.content, str) else str(resp.content)
        except Exception as e:
            logger.warning(f"[memory_store]事实抽取失败: {e}")
            return []
        facts = []
        for line in content.split("\n"):
            line = line.strip().lstrip("-").strip()
            if not line or line.startswith("无") or len(line) < 4 or len(line) > 200:
                continue
            facts.append(line)
        return facts

    def _decide_update(self, new_fact: str, old_fact: str) -> str:
        """更新阶段：用 LLM 判断新事实相对已有事实应执行的操作。"""
        try:
            resp = self.model.invoke(UPDATE_PROMPT.format(new_fact=new_fact, old_fact=old_fact))
            content = resp.content if isinstance(resp.content, str) else str(resp.content)
            op = content.strip().upper()
            for k in ("ADD", "UPDATE", "MERGE", "DELETE"):
                if k in op:
                    return k
        except Exception as e:
            logger.warning(f"[memory_store]更新决策失败，默认 ADD: {e}")
        return "ADD"

    def _search_existing(self, fact: str, k: int = 1) -> list[str]:
        """在已有事实中检索最相似的 k 条（返回事实文本）。"""
        if not self._load_facts():
            return []
        try:
            docs = self.vector_store.similarity_search(fact, k=k)
            return [d.page_content for d in docs]
        except Exception as e:
            logger.warning(f"[memory_store]已有事实检索失败: {e}")
            return []

    def _apply(self, op: str, new_fact: str, existing: list[dict], old_fact_text: str) -> bool:
        """根据操作类型更新 existing（原地修改），返回是否发生变化。"""
        old_idx = -1
        for i, e in enumerate(existing):
            if e["fact"] == old_fact_text:
                old_idx = i
                break
        if op == "ADD":
            existing.append({"fact": new_fact, "ts": time.time()})
            return True
        if op == "UPDATE" and old_idx >= 0:
            existing[old_idx]["fact"] = new_fact
            existing[old_idx]["ts"] = time.time()
            return True
        if op == "MERGE" and old_idx >= 0:
            existing[old_idx]["fact"] = existing[old_idx]["fact"] + "；" + new_fact
            existing[old_idx]["ts"] = time.time()
            return True
        # DELETE：丢弃新事实，不变
        return False

    def _rebuild_vector(self, existing: list[dict]):
        """全量重建向量索引（v1 简化实现：drop 后重新插入）。"""
        try:
            from pymilvus import MilvusClient
            client = MilvusClient(uri=self.uri)
            if client.has_collection(self.collection):
                client.drop_collection(self.collection)
            client.close()
        except Exception as e:
            logger.warning(f"[memory_store]drop collection 失败，将直接重建: {e}")
        self.vector_store = self._build_vector_store()
        if existing:
            docs = [
                Document(page_content=e["fact"], metadata={"ts": float(e.get("ts", 0))})
                for e in existing
            ]
            try:
                self.vector_store.add_documents(docs)
            except Exception as e:
                logger.warning(f"[memory_store]向量重建插入失败: {e}")

    def add(self, messages: list[BaseMessage], user_id: str):
        """提取 + 更新：从 messages 抽事实，去重合并后入库。user_id 预留做多用户隔离。"""
        facts = self._extract_facts(messages)
        if not facts:
            return
        existing = self._load_facts()
        changed = False
        for new_fact in facts:
            top = self._search_existing(new_fact, k=1)
            old_fact_text = top[0] if top else ""
            op = self._decide_update(new_fact, old_fact_text) if old_fact_text else "ADD"
            if self._apply(op, new_fact, existing, old_fact_text):
                changed = True
        if changed:
            self._rebuild_vector(existing)
            self._save_facts(existing)
            logger.info(f"[memory_store]提取 {len(facts)} 条事实，现有 {len(existing)} 条")

    def search(self, query: str, k: int = 3) -> list[str]:
        """检索阶段：按 query 召回 top-K 相关事实，注入下一轮 prompt。"""
        if not self._load_facts():
            return []
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            return [d.page_content for d in docs]
        except Exception as e:
            logger.warning(f"[memory_store]检索失败: {e}")
            return []


if __name__ == "__main__":
    store = MemoryStore()
    demo = [
        HumanMessage(content="order-service 今天 5xx 错误率升高，根因是数据库连接池耗尽"),
        AIMessage(content="已定位：order-service 5xx 升高由连接池耗尽导致，建议扩容连接池。"),
    ]
    store.add(demo, user_id="demo")
    print("召回:", store.search("order-service 5xx", k=3))
