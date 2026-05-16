"""viking L0 向量索引：目录级摘要与条目级摘要的向量化索引。

对应方案文档 §4.2/§4.5：存入 Milvus 的不是完整记忆，而是每条记忆的 L0 向量。
检索第五层的 initial positioning（扫目录级 L0）和 refined exploration
（目录内扫条目级 L0）都建立在这个索引上。

索引里混存两类文本，用 metadata["level"] 区分：
- level="dir"   目录级 L0，即 .abstract.md 的内容
- level="entry" 条目级 L0，即每条记忆的一句话摘要
"""
from __future__ import annotations

from typing import Callable, Optional

from langchain_core.documents import Document

from rag.vector_store import VectorStoreBackend
from utils.logger_handler import logger

COLLECTION = "viking_l0"


class L0Index:
    """L0 向量索引抽象。"""

    def upsert(self, docs: list[Document]) -> None:
        raise NotImplementedError

    def search(
        self, query: str, k: int, level: Optional[str] = None, category: Optional[str] = None
    ) -> list[tuple[Document, float]]:
        """检索。level/category 为 None 时不过滤。返回 [(Document, 余弦分数)]，降序。"""
        raise NotImplementedError


class VectorStoreL0Index(L0Index):
    """基于向量库 adapter 的 L0 索引（生产用，走 config/vector_store.yml 的 provider）。"""

    def __init__(self, backend: Optional[VectorStoreBackend] = None, collection: str = COLLECTION):
        self.collection = collection
        self.backend = backend or build_backend_with(collection)

    def upsert(self, docs: list[Document]) -> None:
        if not docs:
            return
        self.backend.add_documents(docs)

    def search(self, query: str, k: int, level=None, category=None) -> list[tuple[Document, float]]:
        # 多取一些再本地过滤，避免后过滤把结果截断到 0
        hits = self.backend.similarity_search_with_score(query, k=max(k * 4, 8))
        out = []
        for doc, score in hits:
            md = doc.metadata or {}
            if level and md.get("level") != level:
                continue
            if category and md.get("category") != category:
                continue
            out.append((doc, float(score)))
            if len(out) >= k:
                break
        return out


def build_backend_with(collection: str) -> VectorStoreBackend:
    """构造一个独立 collection 的后端实例。

    viking 的 L0 索引与知识库 RAG 共用同一个 Milvus .db 文件，但用独立 collection
    隔离，互不干扰。实现上直接构造另一份 backend 实例、覆写 collection 名即可。
    """
    from rag.vector_store import ChromaBackend, MilvusBackend, vector_conf

    provider = str(vector_conf.get("provider", "milvus")).lower()
    if provider == "chroma":
        return ChromaBackend(collection_name=collection)
    if provider == "milvus":
        return MilvusBackend(collection_name=collection)
    raise ValueError(f"不支持的向量库 provider: {provider}")


def _cos(a: list[float], b: list[float]) -> float:
    import math

    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class InMemoryL0Index(L0Index):
    """进程内 L0 索引，零依赖、可离线跑，用于单测与无向量库环境的降级。

    embedder 由外部注入（默认取项目 embedding 模型）。测试里传一个确定性的
    轻量 embedder 即可完整跑通目录递归检索的下层逻辑。
    """

    def __init__(self, embedder: Optional[Callable[[str], list[float]]] = None):
        if embedder is None:
            from model.factory import embed_model
            embedder = lambda t: list(embed_model.embed_query(t))  # noqa: E731
        self.embedder = embedder
        self._items: list[tuple[list[float], Document]] = []

    def upsert(self, docs: list[Document]) -> None:
        for d in docs:
            try:
                vec = self.embedder(d.page_content)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[l0_index]L0 向量化失败，跳过: {e}")
                continue
            self._items.append((vec, d))

    def search(self, query: str, k: int, level=None, category=None) -> list[tuple[Document, float]]:
        try:
            qv = self.embedder(query)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[l0_index]查询向量化失败: {e}")
            return []
        scored = []
        for vec, doc in self._items:
            md = doc.metadata or {}
            if level and md.get("level") != level:
                continue
            if category and md.get("category") != category:
                continue
            scored.append((doc, _cos(qv, vec)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
