"""viking Directory Recursive Retrieval：目录递归检索算法。

对应方案文档 §4.2 五步与 §4.3 关键设计点。这里把算法完整实现一遍：

  1. Intent Analysis   拆成 0-5 个 TypedQuery（只有 search() 做）
  2. Initial Positioning  query embedding → 扫所有**目录级 L0**（.abstract.md）向量
                         → 阈值过滤，定位多个高分目录
  3. Refined Exploration 进入高分目录 → 扫该目录下**条目级 L0** 向量 → 定位高分条目
  4. Recursive Drill-down 高分条目下钻 L1 概览（默认终点），不够时再下钻 L2
  5. Result Aggregation  按分数与 TypedQuery 优先级聚合，返回最相关的 L1/L2

关键纠正（方案文档 §4.2 明确）：路径不是检索入口，**L0 向量才是入口**；
路径是 L0 命中后告诉你要去哪下钻的导航信息。所以本文件的检索顺序是
"先扫目录级 L0 → 再进目录扫条目级 L0 → 再下钻"，而不是"先猜路径再扫该路径"。

L0 阶段是**阈值过滤而非只取 top-1**：多个目录可能同时相关
（问"磁盘满怎么解决"可能同时命中 incidents 和 solutions）。

全程留 retrieval trace（方案文档 §4.4：viking 强调检索轨迹可视化，
让召回失败可诊断 —— mem0 扁平 top-k 的失败是隐式的，viking 的失败是可观测的）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from utils.logger_handler import logger

from .intent_analyzer import ContextType, IntentAnalyzer, TypedQuery
from .l0_index import L0Index
from .viking_fs import MEMORIES_ROOT, MemoryEntry, VikingFS

# L0 阶段锁定高分目录/条目的余弦阈值
DEFAULT_DIR_THRESHOLD = 0.30
DEFAULT_ENTRY_THRESHOLD = 0.25


@dataclass
class TraceStep:
    """检索轨迹的一 STEP，用于可诊断性回查。"""

    step: str
    detail: str
    hits: int = 0

    def __str__(self) -> str:
        return f"[{self.step}] {self.detail} (命中 {self.hits})"


@dataclass
class RetrievalTrace:
    """检索轨迹：viking 强调 retrieval trace visualization，让召回失败可回溯。"""

    steps: list[TraceStep] = field(default_factory=list)

    def log(self, step: str, detail: str, hits: int = 0) -> None:
        entry = TraceStep(step, detail, hits)
        self.steps.append(entry)
        logger.debug(f"[retrieval] {entry}")

    def render(self) -> str:
        return "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(self.steps))

    @property
    def visited_paths(self) -> list[str]:
        """检索过程中浏览过的虚拟路径，漏召回排查时直接看这个。

        trace 的 detail 形如 "memories/incidents → 扫条目级 L0 命中 2 条"，
        箭头左侧是被浏览的虚拟路径。只收集左侧，按首次出现去重。
        """
        seen: set[str] = set()
        paths: list[str] = []
        for s in self.steps:
            if "→" not in s.detail:
                continue
            head = s.detail.split("→")[0].strip()
            if head and head not in seen:
                seen.add(head)
                paths.append(head)
        return paths


@dataclass
class RetrievedMemory:
    """检索命中的一条记忆及其加载到的层。"""

    entry: MemoryEntry
    layer: str            # L1 默认终点 / L2 深度下钻
    score: float
    path: str
    query_ref: str = ""   # 命中的哪个 TypedQuery

    @property
    def content(self) -> str:
        return self.entry.l2 if self.layer == "L2" else self.entry.l1 or self.entry.l0

    def __str__(self) -> str:
        head = self.content[:80].replace("\n", " ")
        return f"{self.path} [{self.layer} {self.score:.3f}] {head}..."


@dataclass
class RetrievalResult:
    items: list[RetrievedMemory]
    trace: RetrievalTrace
    typed_queries: list[TypedQuery] = field(default_factory=list)
    mode: str = "search"   # search（做意图分析） / find（不做）

    def texts(self, limit: Optional[int] = None) -> list[str]:
        """取出喂给 prompt 的记忆文本。"""
        items = self.items[:limit] if limit else self.items
        return [f"[{it.path} · {it.layer}] {it.content}" for it in items]

    def paths(self) -> list[str]:
        return [it.path for it in self.items]


class DirectoryRecursiveRetriever:
    """viking 目录递归检索：find() 单查询 + search() 多意图检索。"""

    def __init__(
        self,
        vfs: VikingFS,
        index: L0Index,
        intent_analyzer: Optional[IntentAnalyzer] = None,
        dir_threshold: float = DEFAULT_DIR_THRESHOLD,
        entry_threshold: float = DEFAULT_ENTRY_THRESHOLD,
        max_dirs: int = 3,
        detail_depth: int = 2,
    ):
        self.vfs = vfs
        self.index = index
        self.analyzer = intent_analyzer or IntentAnalyzer()
        self.dir_threshold = dir_threshold
        self.entry_threshold = entry_threshold
        self.max_dirs = max_dirs
        # 1=只到 L1（默认终点），2=L1 命中后再下钻 L2
        self.detail_depth = detail_depth

    # ---------- 索引同步 ----------

    def sync_index(self) -> int:
        """把 vfs 里所有目录级 L0 与条目级 L0 写入向量索引。返回索引条目数。"""
        docs = []
        for cat in self.vfs.list_categories():
            ab, _ = self.vfs.read_dir_meta(cat)
            if ab:
                docs.append(
                    self._doc(ab, level="dir", category=cat, path=f"{MEMORIES_ROOT}/{cat}")
                )
        for e in self.vfs.list_entries():
            if e.l0:
                docs.append(
                    self._doc(e.l0, level="entry", category=e.category,
                              entry_id=e.entry_id, path=e.path)
                )
        self.index.upsert(docs)
        return len(docs)

    @staticmethod
    def _doc(text: str, level: str, category: str, path: str, entry_id: str = "") -> "Document":
        from langchain_core.documents import Document
        return Document(
            page_content=text,
            metadata={"level": level, "category": category, "path": path,
                      "entry_id": entry_id},
        )

    # ---------- 五步检索 ----------

    def find(self, query: str, k: int = 3) -> RetrievalResult:
        """简单查询：单 query，不做 intent analysis（省一次 LLM 调用）。"""
        trace = RetrievalTrace()
        trace.log("step0_api", "find() 单查询，跳过 intent analysis")
        trace.log("step1_intent", "跳过", 0)
        preds = [(1.0, ContextType.MEMORY, query)]
        items = self._retrieve_root(preds, k, trace, query)
        return RetrievalResult(items=items, trace=trace, typed_queries=[], mode="find")

    def search(
        self,
        query: str,
        k: int = 3,
        session_summary: str = "",
        last_messages: Optional[list] = None,
    ) -> RetrievalResult:
        """复杂任务：先 intent analysis 拆 0-5 个 TypedQuery，再分目录并行检索。"""
        trace = RetrievalTrace()
        trace.log("step0_api", "search() 多意图检索")

        # 第一~三层：IntentAnalyzer 输出 0-5 个 TypedQuery
        typed = self.analyzer.analyze(query, session_summary, last_messages)
        trace.log("step1_intent", f"拆解出 {len(typed)} 个 TypedQuery", len(typed))

        if not typed:
            trace.log("step2_no_query", "闲聊/无检索意图，直接走 ReAct", 0)
            return RetrievalResult(items=[], trace=trace, typed_queries=typed, mode="search")

        # 第三~四层：按 context_type 分发到不同根目录
        preds: list[tuple[float, ContextType, str]] = []
        for tq in typed:
            priority = max(0.3, min(1.0, tq.priority / 5.0))
            preds.append((priority, tq.context_type, tq.query))

        items = self._retrieve_batch(preds, k, trace, query)
        return RetrievalResult(items=items, trace=trace, typed_queries=typed, mode="search")

    def _retrieve_batch(
        self,
        preds: list[tuple[float, ContextType, str]],
        k: int,
        trace: RetrievalTrace,
        origin_query: str,
    ) -> list[RetrievedMemory]:
        # 并行按 context_type 分发检索
        all_items: list[RetrievedMemory] = []
        for weight, ctx_type, sub_query in preds:
            root = ctx_type.value.lower()          # memories / resources / skills
            hit = self._retrieve_root([(weight, ctx_type, sub_query)], k, trace, sub_query, root)
            all_items.extend(hit)
        # 第五层：结果聚合，按加权分数排序
        all_items.sort(key=lambda it: it.score * self._priority_of(it, preds), reverse=True)
        dedup: list[RetrievedMemory] = []
        seen = set()
        for it in all_items:
            if it.entry.entry_id in seen:
                continue
            seen.add(it.entry.entry_id)
            dedup.append(it)
            if len(dedup) >= k:
                break
        trace.log("step5_aggregate",
                  f"聚合 {len(all_items)} 条候选 → {len(dedup)} 条", len(dedup))
        return dedup

    def _priority_of(self, item: RetrievedMemory, preds) -> float:
        for weight, _ctx, q in preds:
            if q == item.query_ref:
                return weight
        return 1.0

    def _retrieve_root(
        self,
        preds: list[tuple[float, ContextType, str]],
        k: int,
        trace: RetrievalTrace,
        origin_query: str,
        root: str = MEMORIES_ROOT,
    ) -> list[RetrievedMemory]:
        """在某个根目录下执行一次完整目录递归检索（返回未截断的命中）。"""
        items: list[RetrievedMemory] = []
        root_label = root
        for weight, ctx_type, sub_query in preds:
            # ---------- Step 2: Initial Positioning ----------
            # 扫所有目录级 L0 向量，阈值过滤锁定高分目录（复数，不是 top-1）
            dir_hits = self.index.search(sub_query, k=max(self.max_dirs * 2, 6), level="dir")
            locked = [d for d in dir_hits if d[1] >= self.dir_threshold][: self.max_dirs]
            trace.log(
                "step2_initial_positioning",
                f"扫目录级 L0：候选 {len(dir_hits)} 个，"
                f"阈值 {self.dir_threshold} 锁定 {len(locked)} 个目录",
                len(locked),
            )
            if not locked:
                trace.log("step3_miss", f"{root_label} 下无目录命中（L0 摘要可能写得不好）", 0)
                continue

            # ---------- Step 3: Refined Exploration ----------
            for doc, score in locked:
                category = (doc.metadata or {}).get("category", "")
                if not category:
                    continue
                entry_hits = self.index.search(
                    sub_query, k=k * 3, level="entry", category=category
                )
                top = [h for h in entry_hits if h[1] >= self.entry_threshold][:k]
                trace.log(
                    "step3_refined_exploration",
                    f"{MEMORIES_ROOT}/{category} → 扫条目级 L0 命中 {len(top)} 条",
                    len(top),
                )

                # ---------- Step 4: Recursive Drill-down ----------
                for edoc, escore in top:
                    md = edoc.metadata or {}
                    entry = self.vfs.read_entry(md.get("entry_id", ""))
                    if entry is None:
                        continue
                    # L1 是默认终点；detail_depth=2 时再下钻 L2
                    layer = "L1"
                    if self.detail_depth >= 2 and entry.l2 and entry.l2 != entry.l1:
                        layer = "L2"
                        trace.log("step4_drill_down", f"{entry.path} → L2 完整原文", 1)
                    else:
                        trace.log("step4_drill_down", f"{entry.path} → L1 概览（默认终点）", 1)
                    items.append(
                        RetrievedMemory(
                            entry=entry, layer=layer, score=escore * weight,
                            path=entry.path, query_ref=sub_query,
                        )
                    )
        return items
