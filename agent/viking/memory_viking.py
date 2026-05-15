"""viking 长期记忆库：mem0 管"数据进入的治理"，viking 管"数据组织 + 分层取"。

对应方案文档 §2、§3、§5 与附录 A 的写路径：

写路径（commit）：
  1. LLM 从对话流抽取原子事实（mem0 的 fact extraction）
  2. hash 硬去重：完全相同的事实直接丢弃（降写路径 LLM 调用成本）
  3. LLM 分类路由：判断属于哪个子目录（viking 内置分类 + 运维场景扩展）
  4. **限定在该目录内**算相似度（分类在前、相似度比较限定同目录，方案文档 §5.1）
       > 0.92 → 丢弃（重复）
       > 0.85 → LLM 合并：原位更新 L2 + 重生成条目级 L0/L1；分类变了则迁移目录
  5. 同步生成条目级 L0 摘要 / L1 概览（L2 就是原始事实原文）
  6. 给目录打 dirty 标记，目录级 L0/L1 惰性刷新（或定时批量兜底）

读路径见 directory_retrieval.DirectoryRecursiveRetriever。

与 mem0 原生的差异（面试要讲清）：mem0 的 write-path 是 LLM routing 决策
ADD/UPDATE/MERGE/DELETE/NOOP，这里用 hash + 双阈值简化，是为了降低写路径成本。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from agent.viking.directory_retrieval import (
    DEFAULT_DIR_THRESHOLD,
    DEFAULT_ENTRY_THRESHOLD,
    DirectoryRecursiveRetriever,
    RetrievedMemory,
    RetrievalResult,
)
from agent.viking.viking_fs import MEMORIES_ROOT
from agent.viking.intent_analyzer import parse_typed_queries
from agent.viking.l0_index import InMemoryL0Index, L0Index, VectorStoreL0Index
from agent.viking.viking_fs import (
    CATEGORIES,
    DEFAULT_CATEGORY,
    L0_MAX,
    L1_MAX,
    MemoryEntry,
    VikingFS,
)
from model.factory import chat_model
from utils.logger_handler import logger
from utils.path_tool import get_abs_path

# 双阈值（方案文档 §2.2/§2.3，工程简化版）
DEDUP_THRESHOLD = 0.92   # 超过此相似度视为重复，丢弃
MERGE_THRESHOLD = 0.85   # 超过此相似度触发 LLM 合并

CATEGORY_PROMPT = """把下面这条记忆事实归入最合适的一个目录，只输出目录名（小写），不要任何解释。

可选目录：
{categories}

判定参考：
- user_profile：关于用户本人的画像信息（职业、负责服务、技术栈）
- incidents：历史故障（某服务某时层的根因、影响面）
- solutions：验证过的解决方案
- preferences：用户的交互偏好（回复风格、输出格式、关注指标）
- decisions：历史决策及其理由
- misc：以上都不合适时的兜底

事实：{fact}
目录名："""

LAYER_PROMPT = """为下面这条运维记忆生成两层的摘要与概览。

要求：
- L0 一句话摘要：不超过 {l0_max} 字，能独立判断"这条记忆讲什么"，用于向量检索定位
- L1 概览：不超过 {l1_max} 字，包含关键信息、根因、处置手段、适用场景，用于回答用户

记忆原文：
{fact}

按下面格式输出，不要多余文字：
L0: <一句话摘要>
L1: <概览>"""

FACT_EXTRACTION_PROMPT = """你是一个运维记忆事实抽取器。从下面的对话中抽取"值得长期记住的原子事实"，每条一行。
只抽取：用户偏好、决策、长期有效的结论、关键运维结论（如某服务某时段的根因定位）。
不要抽取：临时计算、寒暄、一次性查询、工具中间结果。
若没有值得记住的事实，只输出：无
对话：
{dialog}
事实（每行一条，无编号）："""

MERGE_PROMPT = """把下面两条相似的运维记忆合并成一条更完整的记忆，只输出合并后的内容，不要解释。

已有记忆：{old}
新记忆：{new}
合并后："""


def _hash_fact(fact: str) -> str:
    return hashlib.md5(fact.strip().encode("utf-8")).hexdigest()


class VikingMemoryStore:
    """mem0 治理 + viking 分层组织的长期记忆库。"""

    def __init__(
        self,
        vfs: Optional[VikingFS] = None,
        index: Optional[L0Index] = None,
        retriever: Optional[DirectoryRecursiveRetriever] = None,
        model=None,
        auto_index: bool = True,
        base_dir: Optional[str] = None,
        dir_threshold: Optional[float] = None,
        entry_threshold: Optional[float] = None,
    ):
        self.model = model or chat_model
        if vfs is None:
            vfs = VikingFS(base_dir) if base_dir else VikingFS()
        self.vfs = vfs
        # 去重指纹跟随 vfs 目录，便于按环境/用例隔离
        self.dedup_path = os.path.join(self.vfs.base, "hashes.json")
        self._hashes: set[str] = self._load_hashes()
        # 写路径直线依赖 self.index（commit_fact -> index.upsert），不做兜底的话
        # 不传 index 的调用方（如 ReactAgent 的默认构造）会在每次写记忆时崩，
        # 而异常被上层 except 吞掉只剩一条 warning —— 又一次静默失败
        self.index = index if index is not None else self._default_index()
        # 阈值是可注入的：离线评测/单测用的是假 embedding，余弦尺度跟真实模型不是一把
        # 尺子，必须能显式固定，否则"调阈值"和"调 embedding"会混淆在一起。
        if dir_threshold is None or entry_threshold is None:
            try:
                from utils.config_handler import agent_conf
                dir_threshold = float(
                    agent_conf.get("viking_dir_threshold", DEFAULT_DIR_THRESHOLD)
                    if dir_threshold is None else dir_threshold)
                entry_threshold = float(
                    agent_conf.get("viking_entry_threshold", DEFAULT_ENTRY_THRESHOLD)
                    if entry_threshold is None else entry_threshold)
            except Exception:  # noqa: BLE001 - 配置不可用时退回默认阈值
                if dir_threshold is None:
                    dir_threshold = DEFAULT_DIR_THRESHOLD
                if entry_threshold is None:
                    entry_threshold = DEFAULT_ENTRY_THRESHOLD
        self.dir_threshold = dir_threshold
        self.entry_threshold = entry_threshold
        self.retriever = retriever or DirectoryRecursiveRetriever(
            vfs=self.vfs,
            index=self.index or self._default_index(),
            dir_threshold=dir_threshold,
            entry_threshold=entry_threshold,
        )
        if auto_index:
            self.sync_index()

    def _default_index(self) -> L0Index:
        try:
            return VectorStoreL0Index()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[viking]L0 向量索引不可用，降级为进程内索引: {e}")
            return InMemoryL0Index()

    # ---------- 去重指纹 ----------

    def _load_hashes(self) -> set[str]:
        try:
            with open(self.dedup_path, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except (FileNotFoundError, json.JSONDecodeError):
            return set()

    def _save_hashes(self):
        os.makedirs(os.path.dirname(self.dedup_path), exist_ok=True)
        with open(self.dedup_path, "w", encoding="utf-8") as f:
            json.dump(sorted(self._hashes), f)

    def _seen(self, fact: str) -> bool:
        """hash 硬去重：完全相同的事实不再入库。"""
        h = _hash_fact(fact)
        if h in self._hashes:
            return True
        self._hashes.add(h)
        self._save_hashes()
        return False

    # ---------- LLM 辅助 ----------

    def _llm_text(self, prompt: str) -> str:
        resp = self.model.invoke(prompt)
        content = resp.content if isinstance(resp.content, str) else str(resp.content)
        return content.strip()

    def _extract_facts(self, messages: list[BaseMessage]) -> list[str]:
        dialog = "\n".join(
            f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}"
            for m in messages
            if isinstance(m, (HumanMessage, AIMessage)) and getattr(m, "content", "")
        )
        if not dialog.strip():
            return []
        try:
            content = self._llm_text(FACT_EXTRACTION_PROMPT.format(dialog=dialog))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[viking]事实抽取失败: {e}")
            return []
        facts = []
        for line in content.split("\n"):
            line = line.strip().lstrip("-").strip()
            if not line or line.startswith("无") or len(line) < 4 or len(line) > 200:
                continue
            facts.append(line)
        return facts

    def _route_category(self, fact: str) -> str:
        """分类路由（方案文档 §2.4）：基于 viking 内置分类 + 运维场景扩展。"""
        try:
            content = self._llm_text(
                CATEGORY_PROMPT.format(categories="\n".join(
                    f"- {k}：{v}" for k, v in CATEGORIES.items()), fact=fact)
            )
            name = content.strip().strip("。`\"'").lower()
            # 允许 LLM 输出中文名或带序号，做一次模糊匹配
            for k in CATEGORIES:
                if k in name:
                    return k
            for k, v in CATEGORIES.items():
                if v.split("（")[0][:4] in name or v[:4] in name:
                    return k
            m = re.search(r"[a-z_]+", name)
            if m and m.group() in CATEGORIES:
                return m.group()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[viking]分类路由失败，落兜底目录 misc: {e}")
        return DEFAULT_CATEGORY

    def _generate_layers(self, fact: str) -> tuple[str, str]:
        """生成条目级 L0/L1（方案文档 §3.2：L2 不需要生成，它就是原文）。"""
        try:
            content = self._llm_text(
                LAYER_PROMPT.format(fact=fact, l0_max=L0_MAX, l1_max=L1_MAX))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[viking]L0/L1 生成失败，用原文兜底: {e}")
            head = fact[:L0_MAX]
            return head, fact[:L1_MAX]
        l0, l1 = "", ""
        for line in content.splitlines():
            if line.startswith("L0:"):
                l0 = line[3:].strip()
            elif line.startswith("L1:"):
                l1 = line[3:].strip()
        if not l0:
            l0 = fact[:L0_MAX]
        if not l1:
            l1 = fact[:L1_MAX]
        return l0, l1

    def _merge_facts(self, old: str, new: str) -> str:
        try:
            return self._llm_text(MERGE_PROMPT.format(old=old, new=new))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[viking]合并失败，直接拼接: {e}")
            return f"{old}；{new}"

    # ---------- 写路径 ----------

    def commit_fact(self, fact: str, session_id: str = "") -> Optional[MemoryEntry]:
        """单条事实的完整治理：去重 → 分类 → 同目录相似度 → 合并/新增 → 生成层级。

        返回写入的条目；None 表示被去重丢弃。
        """
        if self._seen(fact):
            logger.info(f"[viking]hash 硬去重命中，跳过: {fact[:40]}")
            return None

        category = self._route_category(fact)
        self.vfs.ensure_category(category)
        l0, l1 = self._generate_layers(fact)

        entry = MemoryEntry(
            entry_id=f"{int(time.time())}_{_hash_fact(fact)[:8]}",
            category=category,
            l0=l0,
            l1=l1,
            l2=fact,
            session_id=session_id,
        )

        # 同目录内相似度比较（方案文档 §5.1：分类在前，相似度比较限定同目录）
        similar = self._search_in_category(fact, category, k=1)
        if similar:
            sim_entry, score = similar[0]
            if score >= DEDUP_THRESHOLD:
                logger.info(f"[viking]软去重命中({score:.3f}>={DEDUP_THRESHOLD})，丢弃: {fact[:40]}")
                return None
            if score >= MERGE_THRESHOLD:
                merged = self._merge_facts(sim_entry.l2 or sim_entry.l1, fact)
                new_l0, new_l1 = self._generate_layers(merged)
                # 合并后可能语义变了 → 重新分类（edge case，方案文档 §5.1 Step 5）
                new_cat = self._route_category(merged)
                updated = self.vfs.update_entry(
                    sim_entry.entry_id, l0=new_l0, l1=new_l1, l2=merged, category=new_cat
                )
                logger.info(
                    f"[viking]合并 {sim_entry.entry_id} → {new_cat} "
                    f"(相似度 {score:.3f}>={MERGE_THRESHOLD})"
                )
                if updated:
                    self.index.upsert([self._entry_doc(updated)])
                    self.vfs.mark_dirty(new_cat)
                return updated

        entry = self.vfs.write(entry)
        self.index.upsert([self._entry_doc(entry)])
        logger.info(f"[viking]新增记忆 {entry.path}")
        return entry

    def commit(self, messages: list[BaseMessage], session_id: str = "") -> list[str]:
        """一轮对话的完整 commit：抽事实 → 逐条治理入库。返回写入的 entry_id 列表。"""
        facts = self._extract_facts(messages)
        written = []
        for fact in facts:
            e = self.commit_fact(fact, session_id=session_id)
            if e is not None:
                written.append(e.entry_id)
        if written:
            logger.info(f"[viking]commit {len(facts)} 条事实 → 入库 {len(written)} 条")
        return written

    def commit_resource(self, doc_id: str, summary: str, content: str,
                        category: str = "resources") -> MemoryEntry:
        """把知识库文档作为 RESOURCE 存入 viking（方案文档 §4.9：viking 统一三类检索源）。"""
        self.vfs.ensure_category(category)
        entry = MemoryEntry(
            entry_id=doc_id,
            category=category,
            l0=summary[:L0_MAX],
            l1=summary[:L1_MAX],
            l2=content,
        )
        entry = self.vfs.write(entry)
        self.index.upsert([self._entry_doc(entry)])
        return entry

    def commit_skill(self, skill_id: str, name: str, usage: str,
                     category: str = "skills") -> MemoryEntry:
        """把工具调用经验作为 SKILL 存入 viking（方案文档 §4.9）。"""
        return self.commit_resource(skill_id, name, usage, category)

    # ---------- 同目录相似度 ----------

    def _entry_doc(self, entry: MemoryEntry):
        from langchain_core.documents import Document
        return Document(
            page_content=entry.l0,
            metadata={"level": "entry", "category": entry.category,
                      "entry_id": entry.entry_id, "path": entry.path},
        )

    def _dir_doc(self, category: str):
        """目录级 L0 文档。目录入口不在索引里，Step2 就扫不到新写入的记忆。"""
        from langchain_core.documents import Document
        abstract, _ = self.vfs.read_dir_meta(category)
        if not abstract:
            return None
        # metadata 的 key 集合必须和条目级文档一致：Milvus 建表时按首次插入的
        # 字段生成 schema，字段非空且无默认值，少插一个 key 就整批 insert 失败。
        # 之前目录级文档没有 entry_id，第一次建表（条目级先写）后目录刷新就一直报错
        return Document(
            page_content=abstract,
            metadata={"level": "dir", "category": category,
                      "entry_id": f"dir:{category}",
                      "path": f"{MEMORIES_ROOT}/{category}"},
        )

    def _search_in_category(self, fact: str, category: str, k: int = 1
                            ) -> list[tuple[MemoryEntry, float]]:
        """在**同一目录内**找最相似的已有条目，返回 [(entry, 分数)]。"""
        try:
            hits = self.index.search(fact, k=max(k * 4, 8), level="entry", category=category)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[viking]同目录相似度检索失败: {e}")
            return []
        out = []
        for doc, score in hits[:k]:
            md = doc.metadata or {}
            entry = self.vfs.read_entry(md.get("entry_id", ""))
            if entry is not None:
                out.append((entry, float(score)))
        return out

    # ---------- 索引同步与目录摘要 ----------

    def sync_index(self) -> int:
        return self.retriever.sync_index()

    def _dir_summary_generator(self):
        """目录级 L0/L1 的生成器：基于目录内真实条目归纳（方案文档 §3.4，惰性刷新）。"""
        def gen(entries: list[MemoryEntry]) -> tuple[str, str]:
            if not entries:
                return "（暂无条目）", ""
            first = [e.l0 for e in entries[:3] if e.l0]
            abstract = (
                f"本目录共 {len(entries)} 条记忆。最新条目摘要："
                + "；".join(first[:2])[:L0_MAX]
            )
            overview = "目录内条目概览：\n" + "\n".join(
                f"- {e.path}：{e.l1 or e.l0}" for e in entries[:10]
            )
            return abstract, overview
        return gen

    def refresh_dirty_dirs(self) -> list[str]:
        """惰性刷新所有 dirty 目录的 L0/L1（检索时按需调用）。

        刷新后必须把新的目录级 L0 一起写回 L0 索引：目录入口是 Step2 唯一的数据源，
        而 sync_index() 是"全量重扫"，写路径不会调它。少了这一步，刚 commit 的记忆
        在下次检索时目录入口还不存在，召回永远为空。
        """
        cats = self.vfs.refresh_all(self._dir_summary_generator())
        if cats and self.index is not None:
            docs = [d for d in (self._dir_doc(c) for c in cats) if d is not None]
            if docs:
                self.index.upsert(docs)
                logger.info(f"[viking]目录级 L0 同步进索引: {len(docs)} 个")
        return cats

    def refresh_dir(self, category: str) -> bool:
        return self.vfs.refresh_dir_meta(category, self._dir_summary_generator())

    # ---------- 读路径 ----------

    def find(self, query: str, k: int = 3) -> RetrievalResult:
        """简单查询：单 query，跳过 intent analysis（省一次 LLM 调用）。"""
        self.refresh_dirty_dirs()
        return self.retriever.find(query, k=k)

    def search(
        self,
        query: str,
        k: int = 3,
        session_summary: str = "",
        last_messages: Optional[list] = None,
        use_intent: Optional[bool] = None,
    ) -> RetrievalResult:
        """默认走 search()（多意图检索）；use_intent=False 时降级为 find() 省一次 LLM。"""
        from agent.viking.intent_analyzer import should_use_search

        if use_intent is None:
            use_intent = should_use_search(query)
        # 检索前惰性刷新脏目录摘要，保证 L0 定位用的是最新目录概览
        self.refresh_dirty_dirs()
        if not use_intent:
            return self.retriever.find(query, k=k)
        return self.retriever.search(query, k=k, session_summary=session_summary,
                                     last_messages=last_messages)

    def recall(self, query: str, k: int = 3) -> list[str]:
        """喂 prompt 用的扁平文本列表（保持与旧 MemoryStore.search 的调用形态一致）。"""
        return self.search(query, k=k).texts()

    def tree(self) -> str:
        return self.vfs.describe_tree()
