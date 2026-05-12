"""
mem0 式长期记忆：轻量实现两阶段流水线，借鉴 mem0 的设计但不引入其完整依赖。

- 提取阶段：每轮对话后用 LLM 从最新交流抽取"值得长期记住的原子事实"
- 更新阶段：新事实向量化检索最相似的已有事实，由 LLM 决定 ADD / UPDATE / MERGE / DELETE
- 检索阶段：下一轮 query 向量化召回 top-K 事实，作为长期记忆注入 prompt

与 mem0 一致采用 ADD-only 倾向：默认追加，矛盾事实并存，靠检索时的元数据时间排序
让新事实在 prompt 中优先呈现。事实原文与元数据以 JSON 文件为 source of truth，
Milvus 向量库作为检索索引。

写入是**增量 upsert**（Roadmap 1 已完成）：`fact_id` 为事实主键，新增只插不重建；
被替换的旧事实按 `fact_id` 差集从向量库删掉，不再 drop collection 全量重插。
注意 viking 开启后（`agent_conf.viking_enabled`）Agent 走 `agent.viking.memory_viking`，
本文件退化为 viking 不可用时的兜底实现。
"""
import hashlib
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


def fact_id_of(fact: str) -> str:
    """事实主键：事实文本做主键，更新后主键随之变化，
    增量 sync 时用差集清理被替换掉的旧向量。"""
    return hashlib.md5(fact.strip().encode("utf-8")).hexdigest()[:16]

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
            existing.append({"fact": new_fact, "ts": time.time(),
                             "fact_id": fact_id_of(new_fact)})
            return True
        if op == "UPDATE" and old_idx >= 0:
            existing[old_idx]["fact"] = new_fact
            existing[old_idx]["ts"] = time.time()
            existing[old_idx]["fact_id"] = fact_id_of(new_fact)
            return True
        if op == "MERGE" and old_idx >= 0:
            existing[old_idx]["fact"] = existing[old_idx]["fact"] + "；" + new_fact
            existing[old_idx]["ts"] = time.time()
            existing[old_idx]["fact_id"] = fact_id_of(existing[old_idx]["fact"])
            return True
        # DELETE：丢弃新事实，不变
        return False

    def _milvus_client(self):
        try:
            from pymilvus import MilvusClient
            return MilvusClient(uri=self.uri)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[memory_store]MilvusClient 不可用，跳过索引差集清理: {e}")
            return None

    def _ids_in_index(self, client) -> set[str]:
        """查询向量库里现存的事实主键，用于增量 sync 的差集计算。"""
        if client is None or not client.has_collection(self.collection):
            return set()
        try:
            rows = client.query(
                self.collection, filter="", output_fields=["fact_id"]
            )
            return {r["fact_id"] for r in rows if r.get("fact_id")}
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[memory_store]读取索引主键失败: {e}")
            return set()

    def _sync_vector(self, existing: list[dict]):
        """增量 upsert（Roadmap 1）。

        v1 是全量 drop + 重建，事实库一大就变成纯开销。这里改成：
        - 以 fact_id 为主键，只插入"索引里没有"的新事实（Milvus add_documents 按主键覆盖，天然 upsert）
        - 用差集删掉"断言里已不存在"的旧向量，保证索引与 JSON 事实库一致
        - 全程不 drop collection
        """
        for e in existing:
            e.setdefault("fact_id", fact_id_of(e["fact"]))

        client = self._milvus_client()
        ids_in_index = self._ids_in_index(client)

        to_add = [e for e in existing if e["fact_id"] not in ids_in_index]
        to_delete = [i for i in ids_in_index if i not in {e["fact_id"] for e in existing}]

        if to_delete and client is not None:
            try:
                expr = f'fact_id in [{",".join(to_delete)}]'
                client.delete(self.collection, filter=expr)
                logger.info(f"[memory_store]增量清理 {len(to_delete)} 条过期向量")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[memory_store]过期向量清理失败: {e}")

        if to_add:
            docs = [
                Document(page_content=e["fact"],
                         metadata={"fact_id": e["fact_id"], "ts": float(e.get("ts", 0))})
                for e in to_add
            ]
            try:
                self.vector_store.add_documents(docs)
                logger.info(f"[memory_store]增量写入 {len(to_add)} 条向量（未重建 collection）")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[memory_store]增量写入向量失败: {e}")

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
            self._sync_vector(existing)
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
