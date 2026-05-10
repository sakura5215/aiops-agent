"""
viking 虚拟文件系统：把记忆条目按 viking 的目录模型组织起来。

对应方案文档 §3.1 ~ §3.4：

- 路径结构根固定为 `memories/`，子目录在 viking 内置记忆类型分类
  （profile / preferences / events / experiences / skills ...）基础上按运维场景扩展。
- 条目级三层：L0 一句话摘要（<=256 字符，进向量索引做定位）、
  L1 概览（<=4000 字符，默认下钻终点）、L2 完整原文。
- 目录级两层：`.abstract.md`（目录级 L0，描述该目录存什么）+ `.overview.md`（目录级 L1）。
- 目录级摘要的更新时机采用"写入时打 dirty 标记 + 检索时惰性刷新 + 定时兜底"，
  这是基于写路径成本的工程推断，不是 viking 既定机制（方案文档 §3.4 已标注）。

持久化：本地目录 + index.json 模拟虚拟文件系统。每个条目一个 JSON 文件，
目录级摘要是两个 .md 文件，dirty 集合落在 index.json 的 `_dirty` 字段。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

from utils.path_tool import get_abs_path
from utils.logger_handler import logger

# ---------- viking 路径结构 ----------

VIKING_ROOT = "memory_fs"
MEMORIES_ROOT = "memories"

# viking 内置记忆类型分类 + 按运维场景扩展
CATEGORIES: dict[str, str] = {
    "user_profile": "用户画像（运维工程师、关注的技术栈、负责的服务）",
    "incidents": "历史故障（某服务某时段的根因定位、影响面）",
    "solutions": "验证过的解决方案（可直接复用的处置方案）",
    "preferences": "用户交互偏好（回复风格、输出格式、关注指标）",
    "decisions": "历史决策与理由（为什么选了某个方案）",
    "misc": "兜底目录：分类路由不确定时暂存，待 review 后迁移",
}
DEFAULT_CATEGORY = "misc"

# 目录级摘要文件（viking 文档明确：每个目录有自己的 abstract / overview）
ABSTRACT_FILE = ".abstract.md"
OVERVIEW_FILE = ".overview.md"

# 条目级三层上限
L0_MAX = 256
L1_MAX = 4000


@dataclass
class MemoryEntry:
    """一个记忆条目，自带 viking 三层。"""

    entry_id: str
    category: str
    l0: str                       # L0 Abstract：一句话摘要，进向量索引
    l1: str = ""                  # L1 Overview：概览，默认下钻终点
    l2: str = ""                  # L2 Detail：完整原文，按需下钻
    ts: float = field(default_factory=time.time)
    session_id: str = ""          # 来源会话，便于回溯
    tags: list[str] = field(default_factory=list)

    @property
    def path(self) -> str:
        """viking 虚拟路径：memories/{category}/{entry_id}"""
        return f"{MEMORIES_ROOT}/{self.category}/{self.entry_id}"

    @property
    def dir_path(self) -> str:
        return f"{MEMORIES_ROOT}/{self.category}"


class VikingFS:
    """viking 虚拟文件系统：条目的目录组织、三层内容、目录级摘要与 dirty 标记。"""

    def __init__(self, base_dir: Optional[str] = None):
        self.base = get_abs_path(base_dir) if base_dir else get_abs_path(VIKING_ROOT)
        self.root_dir = os.path.join(self.base, MEMORIES_ROOT)
        self.index_path = os.path.join(self.base, "index.json")
        os.makedirs(self.root_dir, exist_ok=True)
        self._index: dict = self._load_index()

    # ---------- 索引 ----------

    def _load_index(self) -> dict:
        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "_dirty" in data and "_entries" in data:
                return data
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        return {"_entries": {}, "_dirty": []}

    def _save_index(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        # 保证 _entries / _dirty 字段始终存在
        self._index.setdefault("_entries", {})
        self._index.setdefault("_dirty", [])
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self._index, f, ensure_ascii=False, indent=2)

    @property
    def entries_index(self) -> dict[str, dict]:
        return self._index.setdefault("_entries", {})

    # ---------- 路径 ----------

    def _category_dir(self, category: str) -> str:
        return os.path.join(self.root_dir, category)

    def _entry_file(self, entry_id: str, category: str) -> str:
        return os.path.join(self._category_dir(category), f"{entry_id}.json")

    def ensure_category(self, category: str, seed: bool = True) -> str:
        """确保目录存在。seed=True 时若目录尚无摘要，写入一条"空目录摘要"。

        viking 文档只说写入时自动处理三层，没说空目录怎么办。这里取通用工程实践：
        目录刚建好时用目录名和设计意图生成初始摘要，等有真实条目进来后再刷新。
        """
        d = self._category_dir(category)
        os.makedirs(d, exist_ok=True)
        if seed and not self.read_dir_meta(category)[0]:
            desc = CATEGORIES.get(category, "未分类记忆")
            self.write_dir_meta(
                category,
                abstract=f"本目录：{desc}。",
                overview=f"目录 `{category}` 用于存放{desc}，条目按写入时间命名，"
                         f"每条含 L0 摘要 / L1 概览 / L2 原文三层。",
            )
        return d

    # ---------- 条目读写 ----------

    def write(self, entry: MemoryEntry, ensure_dir: bool = True) -> MemoryEntry:
        """写入条目。返回写入后的条目（l0 可能被裁剪）。"""
        if ensure_dir:
            self.ensure_category(entry.category)
        entry.l0 = (entry.l0 or "")[:L0_MAX]
        entry.l1 = (entry.l1 or "")[:L1_MAX]
        path = self._entry_file(entry.entry_id, entry.category)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(entry), f, ensure_ascii=False, indent=2)
        self.entries_index[entry.entry_id] = {
            "category": entry.category,
            "ts": entry.ts,
            "path": entry.path,
        }
        self.mark_dirty(entry.category)
        self._save_index()
        return entry

    def read_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        meta = self.entries_index.get(entry_id)
        if not meta:
            return None
        path = self._entry_file(entry_id, meta.get("category", DEFAULT_CATEGORY))
        return self._read_entry_file(path, meta.get("category", DEFAULT_CATEGORY))

    def _read_entry_file(self, path: str, category: str) -> Optional[MemoryEntry]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            return MemoryEntry(**d)
        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            return None

    def update_entry(
        self,
        entry_id: str,
        l0: Optional[str] = None,
        l1: Optional[str] = None,
        l2: Optional[str] = None,
        category: Optional[str] = None,
    ) -> Optional[MemoryEntry]:
        """原位更新条目内容，并给目录打 dirty 标记（方案文档 §5.2 合并路径）。

        l0 为空字符串表示"重生成摘要"，传 None 表示"不动这一层"。
        """
        meta = self.entries_index.get(entry_id)
        if not meta:
            logger.warning(f"[viking_fs]待更新条目不存在: {entry_id}")
            return None
        old_cat = meta.get("category", DEFAULT_CATEGORY)
        new_cat = category or old_cat
        path = self._entry_file(entry_id, old_cat)
        entry = self._read_entry_file(path, old_cat)
        if entry is None:
            return None
        if l0 is not None:
            entry.l0 = l0[:L0_MAX]
        if l1 is not None:
            entry.l1 = l1[:L1_MAX]
        if l2 is not None:
            entry.l2 = l2[:L1_MAX]
        entry.ts = time.time()
        # 分类变化时迁移目录（合并后语义变了的重分类，方案文档 §5.1 Step 5）
        if new_cat != old_cat:
            os.makedirs(self._category_dir(new_cat), exist_ok=True)
            os.replace(path, self._entry_file(entry_id, new_cat))
        entry.category = new_cat
        with open(self._entry_file(entry_id, new_cat), "w", encoding="utf-8") as f:
            json.dump(asdict(entry), f, ensure_ascii=False, indent=2)
        self.entries_index[entry_id] = {
            "category": new_cat,
            "ts": entry.ts,
            "path": entry.path,
        }
        self.mark_dirty(old_cat)
        if new_cat != old_cat:
            self.mark_dirty(new_cat)
        self._save_index()
        return entry

    def move_entry(self, entry_id: str, new_category: str) -> Optional[MemoryEntry]:
        """分类变了：迁移目录（等价于 update_entry 的重分类分支）。"""
        return self.update_entry(entry_id, category=new_category)

    def delete_entry(self, entry_id: str) -> bool:
        meta = self.entries_index.get(entry_id)
        if not meta:
            return False
        cat = meta.get("category", DEFAULT_CATEGORY)
        path = self._entry_file(entry_id, cat)
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError as e:
            logger.warning(f"[viking_fs]删除条目文件失败: {e}")
        self.entries_index.pop(entry_id, None)
        self.mark_dirty(cat)
        self._save_index()
        return True

    def list_entries(self, category: Optional[str] = None) -> list[MemoryEntry]:
        """列出条目。category=None 表示全量。用于构建向量索引与目录内扫描。"""
        out: list[MemoryEntry] = []
        if category:
            cats = [category]
        else:
            cats = sorted({m.get("category") for m in self.entries_index.values()})
            cats = [c for c in cats if c]
        for cat in cats:
            d = self._category_dir(cat)
            if not os.path.isdir(d):
                continue
            for fn in sorted(os.listdir(d)):
                if not fn.endswith(".json"):
                    continue
                e = self._read_entry_file(os.path.join(d, fn), cat)
                if e is not None:
                    out.append(e)
        out.sort(key=lambda e: e.ts)
        return out

    def entry_count(self, category: str) -> int:
        """目录实际条目数，用于判断目录级 L0 摘要是否已过期。"""
        return len(list(self.list_entries(category)))

    # ---------- 目录级 L0 / L1 ----------

    def read_dir_meta(self, category: str) -> tuple[str, str]:
        """返回 (abstract, overview)，不存在时返回空串。"""
        d = self._category_dir(category)
        ab, ov = "", ""
        p = os.path.join(d, ABSTRACT_FILE)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                ab = f.read()
        p = os.path.join(d, OVERVIEW_FILE)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                ov = f.read()
        return ab, ov

    def write_dir_meta(self, category: str, abstract: str, overview: str):
        d = self._category_dir(category)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, ABSTRACT_FILE), "w", encoding="utf-8") as f:
            f.write(abstract)
        with open(os.path.join(d, OVERVIEW_FILE), "w", encoding="utf-8") as f:
            f.write(overview)

    def list_categories(self) -> list[str]:
        if not os.path.isdir(self.root_dir):
            return []
        return sorted(
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        )

    # ---------- dirty 机制（惰性刷新） ----------

    def mark_dirty(self, category: str):
        """写入/更新/删除后给目录打标记，表示目录级摘要可能过期。"""
        dirty = self._index.setdefault("_dirty", [])
        if category not in dirty:
            dirty.append(category)
            self._save_index()

    @property
    def dirty_categories(self) -> list[str]:
        return list(self._index.get("_dirty", []))

    def take_dirty(self) -> list[str]:
        """取出并清空 dirty 集合（检索时惰性触发用）。"""
        dirty = self._index.pop("_dirty", []) or []
        self._index["_dirty"] = []
        self._save_index()
        return dirty

    def refresh_dir_meta(
        self,
        category: str,
        generator: Callable[[list[MemoryEntry]], tuple[str, str]],
        force: bool = False,
    ) -> bool:
        """惰性刷新目录级 L0/L1：用 generator 基于目录内真实条目重生成摘要。

        返回是否真的刷新了。force=False 时，只有目录条目数发生变化（dirty）才刷新。
        """
        if not force and category not in self.dirty_categories:
            return False
        entries = self.list_entries(category)
        abstract, overview = generator(entries)
        self.write_dir_meta(category, abstract, overview)
        d = self._category_dir(category)
        try:
            self._index["_dirty"] = [c for c in self._index.get("_dirty", []) if c != category]
            self._save_index()
        except Exception:  # noqa: BLE001 - 索引写失败不影响摘要本身
            logger.warning(f"[viking_fs]刷新后清除 dirty 标记失败: {category}", exc_info=True)
        logger.info(f"[viking_fs]刷新目录级摘要: {category}（{len(entries)} 条）")
        return True

    def refresh_all(
        self,
        generator: Callable[[list[MemoryEntry]], tuple[str, str]],
    ) -> list[str]:
        """定时批量兜底：刷新所有 dirty 目录，返回刷新的目录名。"""
        refreshed = []
        for cat in self.dirty_categories:
            if self.refresh_dir_meta(cat, generator):
                refreshed.append(cat)
        return refreshed

    # ---------- 诊断 ----------

    def describe_tree(self) -> str:
        """打印虚拟文件系统树形结构，用于可诊断性排查（方案文档 §4.4 检索轨迹）。"""
        lines = [MEMORIES_ROOT + "/"]
        for cat in self.list_categories():
            ab, _ = self.read_dir_meta(cat)
            first = ab.splitlines()[0] if ab else "(无摘要)"
            lines.append(f"├── {cat}/  [{self.entry_count(cat)} 条] {first}")
            d = self._category_dir(cat)
            if os.path.isdir(d):
                for fn in sorted(os.listdir(d)):
                    if fn.endswith(".json"):
                        lines.append(f"│   └── {fn}")
        return "\n".join(lines)
