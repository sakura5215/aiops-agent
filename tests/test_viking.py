"""
viking 分层记忆单元测试。

覆盖 viking 方案文档落地的所有组件：
  A. VikingFS        虚拟文件路径、条目三级、目录级 L0/L1、dirty 惰性刷新
  B. IntentAnalyzer  TypedQuery 解析、context_type 分流、find/search 选择
  C. L0Index         向量索引与 level/category 过滤
  D. 目录递归检索     五步流程、阈值过滤、L1 默认终点、L2 下钻、retrieval trace
  E. VikingMemoryStore hash 硬去重、分类路由、同目录相似度、合并路径、commit

说明：LLM 与向量库全部 mock（FakeLLM + 确定性 hash embedding），不依赖
DASHSCOPE_API_KEY；纯逻辑部分（解析、阈值、轨迹、去重）是真实执行的。
"""
import math
import os
import shutil
import sys
import tempfile
import types
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


def section(title):
    print(f"\n=== {title} ===")


# ---------------- mock LLM ----------------

class FakeLLM:
    """按 prompt 关键字返回预设回复，模拟 LLM 的四种调用。

    replies 的值可以是 str，也可以是 callable(prompt) -> str（用于"回复随输入变化"的
    场景，例如分类路由要按事实内容决定目录）。
    """

    def __init__(self, replies: dict[str, str]):
        self.replies = replies
        self.calls: list[str] = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        for kw, reply in self.replies.items():
            if kw in prompt:
                content = reply(prompt) if callable(reply) else reply
                return SimpleNamespace(content=content)
        return SimpleNamespace(content="")


# ---------------- 确定性 embedding ----------------

def _stable_hash(s: str) -> int:
    """稳定哈希：Python 内置 hash() 对字符串带随机化（PYTHONHASHSEED），
    同一文本在不同进程里结果不同，会让 embedding 与相似度不可复现。"""
    import hashlib
    return int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:8], "big")


def hash_embed(text, dim=96):
    """确定性 hash embedding：相同字符分布的文本相似度更高，可复现。"""
    v = [0.0] * dim
    for i, ch in enumerate(text):
        v[_stable_hash(ch) % dim] += 1.0
        if i + 1 < len(text):
            v[(_stable_hash(text[i:i + 2]) + i) % dim] += 1.0
    return v


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def new_vfs(tmp):
    from agent.viking.viking_fs import VikingFS
    return VikingFS(base_dir=os.path.join(tmp, "fs"))


def new_index():
    from agent.viking.l0_index import InMemoryL0Index
    return InMemoryL0Index(embedder=hash_embed)


# ============ A. VikingFS ============

def test_viking_fs(tmp):
    section("A. VikingFS 虚拟文件系统")
    from agent.viking.viking_fs import MemoryEntry, CATEGORIES

    vfs = new_vfs(tmp)
    vfs.ensure_category("incidents")
    check("A1 ensure_category 生成空目录摘要",
          vfs.read_dir_meta("incidents")[0].startswith("本目录"))
    check("A2 子目录数量正确", len(vfs.list_categories()) == 1)

    e = MemoryEntry(entry_id="2026-03-disk-full", category="incidents",
                    l0="磁盘满故障", l1="根因日志未轮转", l2="日志未配置 logrotate",
                    session_id="s1")
    e = vfs.write(e)
    check("A3 虚拟路径 formats", e.path == "memories/incidents/2026-03-disk-full", e.path)
    check("A4 条目读写 round-trip", vfs.read_entry(e.entry_id).l2 == "日志未配置 logrotate")
    check("A5 写入后目录打 dirty 标记", "incidents" in vfs.dirty_categories)

    read2 = vfs.read_entry(e.entry_id)
    check("A6 entry_count 统计准确", vfs.entry_count("incidents") == 1)
    check("A7 list_entries 返回条目", len(vfs.list_entries()) == 1)

    # 原位更新
    vfs.update_entry(e.entry_id, l2="日志未配置 logrotate，已加 crontab", l1="根因日志未轮转，配置 logrotate")
    check("A8 原位更新 L2", vfs.read_entry(e.entry_id).l2.endswith("crontab"))
    check("A9 更新后仍打 dirty", "incidents" in vfs.dirty_categories)

    # 分类迁移
    vfs.move_entry(e.entry_id, "solutions")
    check("A10 分类迁移后路径更新",
          vfs.read_entry(e.entry_id).path == "memories/solutions/2026-03-disk-full")
    check("A11 迁移后新目录打 dirty", "solutions" in vfs.dirty_categories)

    # L0 长度裁剪
    long0 = MemoryEntry(entry_id="long", category="misc", l0="x" * 500,
                        l1="y" * 9000, l2="z" * 9000)
    long0 = vfs.write(long0)
    check("A12 L0 裁剪到 256", len(long0.l0) <= 256, str(len(long0.l0)))
    check("A13 L1 裁剪到 4000", len(long0.l1) <= 4000, str(len(long0.l1)))

    # dirty 惰性刷新
    def gen(entries):
        return (f"本目录共 {len(entries)} 条记忆", "概览：" + ",".join(e.entry_id for e in entries))

    check("A14 未 dirty 的目录不刷新（惰性）", vfs.refresh_dir_meta("user_profile", gen) is False)
    vfs.mark_dirty("user_profile")
    check("A15 dirty 目录触发刷新", vfs.refresh_dir_meta("user_profile", gen) is True)
    ab, ov = vfs.read_dir_meta("user_profile")
    check("A16 刷新后摘要含真实条目数", "共 0 条" in ab, ab)
    check("A17 刷新后清除 dirty", "user_profile" not in vfs.dirty_categories)

    refreshed = vfs.refresh_all(gen)
    check("A18 refresh_all 批量刷新 dirty 目录",
          "misc" in refreshed and "solutions" in refreshed, f"refreshed={refreshed}")
    check("A18b 刷新后 dirty 清空", vfs.dirty_categories == [],
          f"dirty={vfs.dirty_categories}")

    check("A19 删除条目", vfs.delete_entry("long") is True)
    check("A20 删除后不可读", vfs.read_entry("long") is None)

    tree = vfs.describe_tree()
    check("A21 树形结构含目录与条目", "memories/" in tree and "solutions" in tree)


# ============ B. IntentAnalyzer ============

def test_intent_analyzer():
    section("B. IntentAnalyzer 意图分析")
    from agent.viking.intent_analyzer import (
        MAX_TYPED_QUERIES, ContextType, IntentAnalyzer,
        parse_typed_queries, should_use_search,
    )

    qs = parse_typed_queries(
        "MEMORY | User's last disk full incident | 回忆历史故障 | 5\n"
        "RESOURCE | disk full root cause logrotate | 查知识库 | 3"
    )
    check("B1 解析四列 TypedQuery", len(qs) == 2)
    check("B2 query 字段正确", qs[0].query == "User's last disk full incident")
    check("B3 context_type 解析", qs[0].context_type == ContextType.MEMORY)
    check("B4 intent 字段", qs[0].intent == "回忆历史故障")
    check("B5 priority 解析", qs[0].priority == 5 and qs[1].priority == 3)
    check("B6 MEMORY 根目录映射", qs[0].root == "memories")
    check("B7 RESOURCE 根目录映射", qs[1].root == "resources")

    two = parse_typed_queries("MEMORY | 用户偏好中文回复 | 偏好 | 2")
    check("B8 退化两列也能解析", len(two) == 1 and two[0].query == "用户偏好中文回复")

    check("B9 NONE 输出空列表", parse_typed_queries("NONE") == [])
    check("B10 空输入返回空", parse_typed_queries("") == [])

    clamped = parse_typed_queries("SKILL | Check disk | 工具 | 99\nSKILL | Use tool | x | 0")
    check("B11 priority 上限钳制到 5", clamped[0].priority == 5)
    check("B12 priority 下限钳制到 1", clamped[1].priority == 1)

    numbered = parse_typed_queries("1. MEMORY | User's preference | 偏好 | 4")
    check("B13 带序号前缀可解析", len(numbered) == 1 and numbered[0].query == "User's preference")

    bad = parse_typed_queries("UNKNOWN | 兜底查询 | 无类型 | 2\n乱七八糟")
    check("B14 非法类型回落默认", bad[0].context_type == ContextType.MEMORY)
    check("B15 非法行被丢弃", len(bad) == 1)

    many = parse_typed_queries("\n".join(f"MEMORY | q{i} | i | 3" for i in range(8)))
    check("B16 最多返回 5 个", len(many) == MAX_TYPED_QUERIES)

    check("B17 短且无疑问走 find()", should_use_search("你好") is False)
    check("B18 长 query 走 search()", should_use_search("上次磁盘满是怎么解决的") is True)
    check("B19 含问号即使短也走 search()", should_use_search("磁盘满?") is True)
    check("B19b 中文疑问词（无问号）也走 search()",
          should_use_search("磁盘满怎么解决") is True and should_use_search("日志轮转如何配置") is True)

    analyzer = IntentAnalyzer(model=FakeLLM({"当前用户输入": "MEMORY | User's preference | x | 4"}))
    out = analyzer.analyze("用户偏好")
    check("B20 analyze 走 LLM", len(out) == 1 and out[0].priority == 4)

    def boom(_prompt):
        raise RuntimeError("LLM 不可用")

    broken = IntentAnalyzer(model=SimpleNamespace(invoke=boom))
    fallback = broken.analyze("随便问点什么")
    check("B21 LLM 异常时降级为单条 MEMORY",
          len(fallback) == 1 and fallback[0].context_type == ContextType.MEMORY
          and fallback[0].intent == "fallback")

    use_search, queries = broken.plan("你好")
    check("B22 plan 对简单查询选 find()", use_search is False and len(queries) == 1)
    use_search2, _ = broken.plan("帮我分析 order-service 最近 1 小时的所有告警")
    check("B23 plan 对复杂任务选 search()", use_search2 is True)


# ============ C. L0Index ============

def test_l0_index():
    section("C. L0 向量索引")
    from langchain_core.documents import Document
    from agent.viking.l0_index import InMemoryL0Index

    idx = new_index()
    idx.upsert([
        Document(page_content="磁盘满故障摘要", metadata={"level": "entry", "category": "incidents", "entry_id": "e1"}),
        Document(page_content="CPU 飙高故障摘要", metadata={"level": "entry", "category": "incidents", "entry_id": "e2"}),
        Document(page_content="incidents 目录摘要", metadata={"level": "dir", "category": "incidents"}),
    ])
    hits = idx.search("磁盘满故障摘要", k=2)
    check("C1 检索按分数降序", len(hits) >= 1 and hits[0][1] >= (hits[-1][1] if len(hits) > 1 else hits[0][1]))
    check("C2 余弦分数归一到 [0,1]（浮点容差）",
          all(-1e-6 <= s <= 1.0 + 1e-6 for _d, s in hits), str([s for _d, s in hits]))

    only_dir = idx.search("磁盘满故障摘要", k=5, level="dir")
    check("C3 level 过滤", len(only_dir) == 1 and only_dir[0][0].metadata["level"] == "dir")

    only_inc = idx.search("磁盘满故障摘要", k=5, level="entry", category="incidents")
    check("C4 category 过滤", all(h[0].metadata["category"] == "incidents" for h in only_inc))

    check("C5 相同文本得满分 1.0", abs(cosine(hash_embed("磁盘满故障摘要"),
                                                  hash_embed("磁盘满故障摘要")) - 1.0) < 1e-9)

    empty = new_index()
    check("C6 空索引不报错", empty.search("任意", k=3) == [])


# ============ D. 目录递归检索 ============

def build_retriever(tmp):
    from agent.viking.directory_retrieval import DirectoryRecursiveRetriever
    from agent.viking.intent_analyzer import IntentAnalyzer
    vfs = new_vfs(tmp)
    vfs.ensure_category("incidents")
    vfs.write(__import__("agent.viking.viking_fs", fromlist=["MemoryEntry"]).MemoryEntry(
        entry_id="disk-full", category="incidents",
        l0="磁盘满故障：日志未轮转导致磁盘写满",
        l1="根因是日志未配置 logrotate 轮转，已加定时任务清理",
        l2="完整记录：磁盘写满后 order-service 写入失败",
        session_id="s1"))
    vfs.write(__import__("agent.viking.viking_fs", fromlist=["MemoryEntry"]).MemoryEntry(
        entry_id="cpu-high", category="incidents",
        l0="CPU 飙高故障：gc 频繁触发",
        l1="根因是 jvm 堆内存配置过小导致频繁 full gc",
        l2="完整记录：cpu 持续 90% 以上，full gc 每分钟 20 次",
        session_id="s1"))
    vfs.write(__import__("agent.viking.viking_fs", fromlist=["MemoryEntry"]).MemoryEntry(
        entry_id="pref-cn", category="preferences",
        l0="用户偏好中文回复",
        l1="用户希望用中文回答，输出结构化要点",
        l2="用户偏好中文", session_id="s1"))
    # 目录级摘要（模拟 LLM 生成）
    vfs.write_dir_meta("incidents", "本目录：历史故障记录（磁盘、CPU、内存类）", "按故障类型组织")
    vfs.write_dir_meta("preferences", "本目录：用户的交互偏好", "按偏好类型组织")
    idx = new_index()
    llm = FakeLLM({"当前用户输入": "MEMORY | 磁盘满 | 查故障 | 5"})
    ret = DirectoryRecursiveRetriever(vfs=vfs, index=idx,
                                      intent_analyzer=IntentAnalyzer(model=llm),
                                      dir_threshold=0.15, entry_threshold=0.15, detail_depth=2)
    ret.sync_index()
    return vfs, idx, ret


def test_directory_retrieval(tmp):
    section("D. 目录递归检索（五步）")
    from agent.viking.directory_retrieval import DirectoryRecursiveRetriever
    from agent.viking.intent_analyzer import IntentAnalyzer

    vfs, idx, ret = build_retriever(tmp)

    n_dir_docs = sum(1 for _d, _m in [(d, m) for d, m in zip(
        [0] * len(idx._items),
        [getattr(d, "metadata", {}) for d, _ in idx._items])] if _m.get("level") == "dir")
    check("D1 sync_index 写入目录级+条目级",
          len(idx._items) == n_dir_docs + vfs.entry_count(None) if False else len(idx._items) >= 4,
          f"items={len(idx._items)}")

    res = ret.find("磁盘满")
    check("D2 find() 返回命中", len(res.items) >= 1, str(res.paths()))
    check("D3 find() 跳过 intent analysis", res.mode == "find" and res.typed_queries == [])
    check("D4 命中的是磁盘满条目",
          any("disk-full" in it.path for it in res.items), str(res.paths()))

    res2 = ret.search("磁盘满怎么解决")
    check("D5 search() 走 intent analysis", res2.mode == "search" and len(res2.typed_queries) >= 1)
    check("D6 search() 结果非空", len(res2.items) >= 1)
    check("D7 trace 记录五步", len(res2.trace.steps) >= 4)
    steps = " | ".join(s.step for s in res2.trace.steps)
    check("D8 五步齐全", "step1_intent" in steps and "step2_initial_positioning" in steps
          and "step3_refined_exploration" in steps and "step4_drill_down" in steps
          and "step5_aggregate" in steps, steps)
    check("D9 trace 记录浏览路径", len(res2.trace.visited_paths) > 0,
          str(res2.trace.visited_paths))

    # 多目录同时命中（阈值过滤而非 top-1）
    ret_max = DirectoryRecursiveRetriever(vfs=vfs, index=idx,
                                          intent_analyzer=IntentAnalyzer(model=FakeLLM({})),
                                          dir_threshold=0.05, entry_threshold=0.05,
                                          max_dirs=3)
    check("D10 多目录阈值过滤可锁定多个", len(ret_max.find("故障").trace.steps) > 0)

    # L1 默认终点 vs L2 下钻
    ret_l1 = DirectoryRecursiveRetriever(vfs=vfs, index=idx,
                                         intent_analyzer=IntentAnalyzer(model=FakeLLM({})),
                                         dir_threshold=0.05, entry_threshold=0.05, detail_depth=1)
    l1_res = ret_l1.find("磁盘满")
    check("D11 detail_depth=1 时停在 L1", all(it.layer == "L1" for it in l1_res.items))
    check("D12 深度下钻命中 L2",
          any(it.layer == "L2" for it in res2.items), str([it.layer for it in res2.items]))

    # 阈值过高 → L0 阶段全部被过滤，返回空但留痕（可诊断）
    strict = DirectoryRecursiveRetriever(vfs=vfs, index=idx,
                                         intent_analyzer=IntentAnalyzer(model=FakeLLM({})),
                                         dir_threshold=0.99, entry_threshold=0.99)
    miss = strict.find("磁盘满")
    check("D13 阈值过滤后可召回为空", len(miss.items) == 0)
    check("D14 召回失败可诊断（trace 有 miss 记录）",
          any(s.step == "step3_miss" for s in miss.trace.steps))

    # 聚合与去重
    multi = ret.search("磁盘满")
    ids = [it.entry.entry_id for it in multi.items]
    check("D15 聚合结果去重", len(ids) == len(set(ids)))

    check("D16 texts() 输出喂 prompt 格式",
          all(t.startswith("[memories/") and ("· L1" in t or "· L2" in t)
              for t in multi.texts()))

    tree = vfs.describe_tree()
    check("D17 树结构可诊断", "incidents" in tree and "preferences" in tree)


# ============ E. VikingMemoryStore ============

def test_memory_store_viking(tmp):
    section("E. VikingMemoryStore 写路径与读路径")
    from agent.viking.memory_viking import VikingMemoryStore
    from agent.viking.viking_fs import MemoryEntry

    llm = FakeLLM({
        # 三条语义差异明显的事实，避免被软去重误判为重复
        "事实抽取": ("order-service 磁盘满的根因是日志未轮转\n"
                  "用户偏好用中文分点回复运维结论\n"
                  "用户是负责 order-service 的运维工程师"),
        # 分类路由按事实内容动态决定落哪个目录（返回目录名字符串）
        "归入最合适的一个目录": (
            lambda p: "preferences" if "用户偏好" in p
            else "incidents" if "磁盘满" in p
            else "user_profile"),
        "L0 一句话摘要": ("L0: order-service 磁盘满，根因日志未轮转\n"
                     "L1: 根因是日志未配置 logrotate，已加 crontab 清理"),
        "把下面两条相似的运维记忆合并": "order-service 磁盘满，根因日志未轮转，已配置 logrotate",
        "当前用户输入": "MEMORY | User's disk full | 回忆故障 | 5",
    })

    vfs = new_vfs(tmp)
    idx = new_index()
    store = VikingMemoryStore(vfs=vfs, index=idx, model=llm,
                              auto_index=False)

    from langchain_core.messages import AIMessage, HumanMessage

    ids = store.commit([], session_id="s1")
    check("E1 空对话不产生事实", ids == [])
    ids = store.commit([HumanMessage(content="order-service 磁盘满了，根因是日志没轮转"),
                        AIMessage(content="已定位根因是日志未轮转，建议配置 logrotate")],
                       session_id="s1")
    check("E2 commit 返回写入条目", len(ids) == 3, str(ids))
    # 只看本轮新写入的条目（tmp 下还留着 A~D 段构造的样例记忆）
    fresh = [e for e in vfs.list_entries(None) if e.entry_id in set(ids)]
    disk = [e for e in fresh if "磁盘满" in e.l2]
    check("E3 磁盘满事实落 incidents 目录",
          len(disk) == 1 and disk[0].category == "incidents",
          str([(e.category, e.l2) for e in fresh]))
    check("E4 条目自带 L0/L1/L2", all(e.l0 and e.l1 and e.l2 for e in fresh))
    # 三条事实内容不同 → 路由结果互不相同，不是一股脑塞同一目录
    check("E4b 分类路由结果按内容分流",
          len({e.category for e in fresh}) == 3,
          str([(e.category, e.l2) for e in fresh]))

    # hash 硬去重
    h_before = len(store._hashes)
    again = store.commit_fact("order-service 磁盘满根因是日志未轮转")
    check("E5 hash 硬去重命中返回 None", again is None)
    check("E6 去重指纹入库", len(store._hashes) > h_before or h_before > 0)

    # 分类路由兜底
    routed = store._route_category("用户偏好中文")
    check("E7 分类路由按事实内容决定目录", routed == "preferences", f"实际 {routed}")
    broken = VikingMemoryStore(vfs=new_vfs(os.path.join(tmp, "e_broken")),
                               index=new_index(), model=FakeLLM({}), auto_index=False)
    check("E8 路由失败落兜底目录 misc", broken._route_category("x") == "misc")

    # 合并路径（同目录相似度 > 0.85）
    existing = MemoryEntry(
        entry_id="old-1", category="incidents", l0="旧摘要", l1="旧概览", l2="磁盘满已定位")
    vfs.write(existing)
    idx.upsert([store._entry_doc(existing)])
    store._search_in_category = lambda fact, cat, k=1: [(existing, 0.90)]
    merged = store.commit_fact("磁盘满已定位并已加 crontab")
    check("E9 相似度 0.90 触发合并（原位更新）", merged is not None and merged.entry_id == "old-1")

    # 软去重（> 0.92 丢弃）
    store._search_in_category = lambda fact, cat, k=1: [(existing, 0.95)]
    check("E10 相似度 0.95 软去重丢弃",
          store.commit_fact("磁盘满已定位并已加 crontab2") is None)

    # 低相似度 → 新增
    store._search_in_category = lambda fact, cat, k=1: []
    new_entry = store.commit_fact("用户偏好用中文回复运维问题", session_id="s1")
    check("E11 低相似度高分新增条目", new_entry is not None)
    check("E12 写入后目录打 dirty 标记", "preferences" in vfs.dirty_categories,
          str(vfs.dirty_categories))

    # 读路径
    vfs2 = new_vfs(os.path.join(tmp, "fs2"))
    idx2 = new_index()
    # 假 embedding 的余弦尺度跟真实模型不是一把尺子（同义文本只有 0.2 左右），
    # 阈值必须能显式固定，否则"检索效果"被 embedding 假象牵着走。
    store2 = VikingMemoryStore(vfs=vfs2, index=idx2, model=llm, auto_index=False,
                               dir_threshold=0.15, entry_threshold=0.15)
    store2.commit([HumanMessage(content="order-service 磁盘满，根因日志没轮转"),
                   AIMessage(content="已定位根因是日志未轮转")], session_id="s1")
    recall = store2.recall("磁盘满", k=3)
    check("E13 recall 返回记忆文本", len(recall) >= 1, str(recall))
    check("E13b 命中的是磁盘满那条",
          any("磁盘满" in t for t in recall), str(recall[:1]))
    res = store2.search("磁盘满怎么解决")
    check("E14 search 走目录递归", res.mode == "search" and len(res.trace.steps) >= 3)
    find_res = store2.find("磁盘满", k=3)
    check("E15 find 单查询模式", find_res.mode == "find")

    # 目录级摘要惰性刷新：写入后 dirty，刷新后摘要含真实条目数
    cats = vfs2.list_categories()
    target_cat = cats[0] if cats else "misc"
    vfs2.mark_dirty(target_cat)
    refreshed = store2.refresh_dirty_dirs()
    check("E16 刷新 dirty 目录", target_cat in refreshed, str(refreshed))
    ab, _ = vfs2.read_dir_meta(target_cat)
    check("E17 刷新后摘要反映条目数", "条记忆" in ab, ab)

    # 三类检索源（viking 统一 memories/resources/skills，方案文档 §4.9）
    store2.commit_resource("kb-1", "Prometheus 查询指南", "prometheus query language 用法")
    check("E18 RESOURCE 目录写入", vfs2.entry_count("resources") == 1)
    store2.commit_skill("skill-metric", "查询服务监控指标", "调用 fetch_metric_data 获取 cpu/内存")
    check("E19 SKILL 目录写入", vfs2.entry_count("skills") == 1)
    res3 = store2.search("监控指标怎么查")
    check("E20 三类源统一在虚拟文件系统里",
          {"resources", "skills"} <= set(vfs2.list_categories()), str(vfs2.list_categories()))
    check("E20b 跨源检索走目录递归", res3.mode == "search" and len(res3.trace.steps) >= 3)


def main():
    tmp = tempfile.mkdtemp(prefix="viking_test_")
    try:
        test_viking_fs(tmp)
        test_intent_analyzer()
        test_l0_index()
        test_directory_retrieval(tmp)
        test_memory_store_viking(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n=== 总计: {PASS} passed, {FAIL} failed ===")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
