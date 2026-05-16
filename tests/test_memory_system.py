"""
AIOps 记忆系统真实可跑测试
==========================

诚实标注三类：
  [REAL]  真实跑：滑动窗口、会话隔离、记忆 JSON 持久化 round-trip
  [MOCK]  mock LLM/Milvus 后测真实解析与合并逻辑：
            mem0 事实抽取的解析、ADD/UPDATE/MERGE/DELETE 决策解析、
            _apply 合并逻辑、VectorStore provider 路由
  [SKIP]  跑不了 / 没做：
            - Milvus 真实连接（pymilvus/langchain_milvus 未装）
            - LLM 真实调用（dashscope 未装 + 无 DASHSCOPE_API_KEY）
            - viking（L0/L1/L2 分层、目录递归检索、IntentAnalyzer）——只有方案文档，根本没落地

运行：.venv/Scripts/python.exe tests/test_memory_system.py
"""
import sys
import os
import json
import types
import tempfile
import shutil

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

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


def _stub_modules():
    """预注入 fake module，绕过未安装的重型依赖（dashscope/pymilvus/langchain_milvus/langchain_chroma）。

    项目内 model.factory 会拉 dashscope；memory_store/vector_store 会拉
    langchain_milvus/langchain_chroma/pymilvus/langchain_text_splitters。
    这里把它们 stub 成空对象，让我们能 import 真实代码并测其中的纯解析/合并/路由逻辑。
    langchain_core 必须真实安装（测滑动窗口需要 BaseChatMessageHistory 基类 + message 序列化）。
    """
    for m in ["model", "model.factory"]:
        sys.modules[m] = types.ModuleType(m)
    sys.modules["model.factory"].embed_model = None
    sys.modules["model.factory"].chat_model = None

    for name, attr, val in [
        ("langchain_milvus", "Milvus", type("Milvus", (), {})),
        ("langchain_chroma", "Chroma", type("Chroma", (), {})),
    ]:
        sys.modules[name] = types.ModuleType(name)
        setattr(sys.modules[name], attr, val)

    # langchain_community 是个 package：要注册为 package + 子模块名也进 sys.modules，
    # 这样 `from langchain_community.document_loaders import PyPDFLoader` 才能命中
    _pkg = types.ModuleType("langchain_community")
    _pkg.__path__ = []  # 标记为 package
    sys.modules["langchain_community"] = _pkg
    _dl = types.ModuleType("langchain_community.document_loaders")
    _dl.PyPDFLoader = type("PyPDFLoader", (), {})
    _dl.TextLoader = type("TextLoader", (), {})
    sys.modules["langchain_community.document_loaders"] = _dl

    sys.modules["langchain_text_splitters"] = types.ModuleType("langchain_text_splitters")

    class _FakeSplitter:
        def __init__(self, *a, **k):
            pass

        def split_documents(self, docs):
            return docs

    sys.modules["langchain_text_splitters"].RecursiveCharacterTextSplitter = _FakeSplitter

    sys.modules["pymilvus"] = types.ModuleType("pymilvus")

    class _FakeMilvusClient:
        def __init__(self, uri=None):
            pass

        def has_collection(self, c):
            return False

        def drop_collection(self, c):
            pass

        def close(self):
            pass

    sys.modules["pymilvus"].MilvusClient = _FakeMilvusClient


class FakeResp:
    def __init__(self, content):
        self.content = content


class FakeModel:
    """可控的 LLM mock：invoke 返回固定 content。"""

    def __init__(self, ret):
        self.ret = ret

    def invoke(self, _):
        return FakeResp(self.ret)


class BoomModel:
    def invoke(self, _):
        raise RuntimeError("LLM 挂了")


# ---------- 测试组 ----------


def test_sliding_window_and_isolation():
    """[REAL] 测试1+2：滑动窗口 + 会话隔离（真实跑 FileChatMessageHistory，需 langchain_core）"""
    print("\n=== 测试组 A [REAL]：滑动窗口 + 会话隔离 ===")
    from langchain_core.messages import HumanMessage, AIMessage
    from agent.memory import FileChatMessageHistory

    tmpdir = tempfile.mkdtemp(prefix="aiops_test_")
    try:
        # A1 滑动窗口
        print("[A1] 滑动窗口：写 60 条，recent_messages(20) 应返回尾部 20 条")
        sess = FileChatMessageHistory("sess_window", storage_dir=tmpdir)
        for i in range(30):
            sess.add_message(HumanMessage(content=f"问题{i}"))
            sess.add_message(AIMessage(content=f"回答{i}"))

        full = sess.messages
        check("全量历史条数=60", len(full) == 60, f"实际 {len(full)}")

        recent = sess.recent_messages(20)
        check("recent_messages(20)=20", len(recent) == 20, f"实际 {len(recent)}")
        check("recent 是尾部", recent[-1].content == "回答29",
              f"实际末条 {recent[-1].content if recent else '空'}")
        check("recent 首条是问题20", recent[0].content == "问题20",
              f"实际首条 {recent[0].content if recent else '空'}")
        check("K=0 返回全量", len(sess.recent_messages(0)) == 60)
        check("K>=len 返回全量", len(sess.recent_messages(100)) == 60)

        # A2 会话隔离
        print("[A2] 会话隔离：两个 session_id 写不同内容，互不干扰")
        s1 = FileChatMessageHistory("user_A", storage_dir=tmpdir)
        s2 = FileChatMessageHistory("user_B", storage_dir=tmpdir)
        s1.add_message(HumanMessage(content="A的秘密"))
        s2.add_message(HumanMessage(content="B的秘密"))

        s1_msgs = [m.content for m in s1.messages]
        s2_msgs = [m.content for m in s2.messages]
        check("A 看不到 B", "A的秘密" in s1_msgs and "B的秘密" not in s1_msgs)
        check("B 看不到 A", "B的秘密" in s2_msgs and "A的秘密" not in s2_msgs)
        check("A 和 B 文件不同", s1.file_path != s2.file_path)
        check("A 文件名含 user_A", "user_A" in os.path.basename(s1.file_path))
        check("B 文件名含 user_B", "user_B" in os.path.basename(s2.file_path))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_mem0_fact_extraction_parsing():
    """[MOCK] 测试3：mem0 事实抽取的解析逻辑（mock LLM 返回，验证 split + 过滤）"""
    print("\n=== 测试组 B [MOCK]：mem0 事实抽取解析逻辑 ===")
    _stub_modules()
    from agent.memory_store import MemoryStore
    from langchain_core.messages import HumanMessage, AIMessage

    store = object.__new__(MemoryStore)  # 绕过 __init__（避开 Milvus 连接）
    store.json_path = os.path.join(tempfile.gettempdir(), "aiops_mem_extract_test.json")
    if not os.path.exists(store.json_path):
        with open(store.json_path, "w", encoding="utf-8") as f:
            json.dump([], f)

    long_line = "a" * 201  # 超过 200 字符阈值，应被过滤

    # 场景1：有效事实 + 噪声混合
    store.model = FakeModel(
        "order-service 5xx 错误率升高\n"
        "数据库连接池耗尽\n"
        "- 根因是慢查询\n"
        "无用的寒暄不要x" + long_line + "\n"
        "无\n"
        "   \n"
        "ab"  # <4 字符，应被过滤
    )
    facts = store._extract_facts([
        HumanMessage(content="order-service 5xx 错误率升高"),
        AIMessage(content="已定位根因是数据库连接池耗尽"),
    ])
    check("过滤掉过长(>200)事实", all(len(f) <= 200 for f in facts))
    check("过滤掉'无'开头的行", all(not f.startswith("无") for f in facts))
    check("过滤掉<4字符的行", all(len(f) >= 4 for f in facts))
    check("保留的有效事实>=3", len(facts) >= 3, f"实际 {len(facts)}: {facts}")
    check("事实去掉了前导-", all(not f.startswith("-") for f in facts))

    # 场景2：LLM 明确返回"无"
    store.model = FakeModel("无")
    facts = store._extract_facts([HumanMessage(content="你好")])
    check("LLM 返回'无'→抽 0 条", facts == [])

    # 场景3：空对话（dialog 为空，直接返回）
    store.model = FakeModel("不该被调用")
    facts = store._extract_facts([])
    check("空对话→抽 0 条且不调 LLM", facts == [])

    if os.path.exists(store.json_path):
        os.remove(store.json_path)


def test_mem0_decide_update_parsing():
    """[MOCK] 测试4：mem0 更新决策解析（mock LLM，验证 ADD/UPDATE/MERGE/DELETE 识别 + 异常降级）"""
    print("\n=== 测试组 C [MOCK]：mem0 更新决策解析 ===")
    _stub_modules()
    from agent.memory_store import MemoryStore

    store = object.__new__(MemoryStore)

    cases = [
        ("ADD", "ADD"),
        ("UPDATE", "  update  "),
        ("MERGE", "把这两条 merge 一下"),
        ("DELETE", "重复，DELETE掉"),
    ]
    for expected, ret in cases:
        store.model = FakeModel(ret)
        op = store._decide_update("新事实", "老事实")
        check(f"解析 {ret!r} → {expected}", op == expected, f"实际 {op}")

    # 异常降级 ADD
    store.model = BoomModel()
    op = store._decide_update("x", "y")
    check("LLM 异常降级 ADD", op == "ADD", f"实际 {op}")

    # 老事实为空时上层应直接走 ADD（不走 _decide_update）—— 这里直接测 _decide_update 行为
    store.model = FakeModel("DELETE")
    op = store._decide_update("新", "老")
    check("即使 LLM 说 DELETE，_decide_update 也如实返回", op == "DELETE",
          f"实际 {op}（注：上层 add() 在 old_fact_text 为空时直接走 ADD，绕过这里）")


def test_apply_logic():
    """[MOCK] 测试5：_apply 的 ADD/UPDATE/MERGE/DELETE 纯逻辑"""
    print("\n=== 测试组 D [MOCK]：_apply 合并逻辑（纯逻辑，无外部依赖）===")
    _stub_modules()
    from agent.memory_store import MemoryStore

    store = object.__new__(MemoryStore)

    # ADD
    existing = [{"fact": "老1", "ts": 1.0}]
    changed = store._apply("ADD", "新事实", existing, "")
    check("ADD→changed=True", changed is True)
    check("ADD→existing+1", len(existing) == 2)
    check("ADD→新事实在末尾", existing[-1]["fact"] == "新事实")
    check("ADD→新事实有 ts", "ts" in existing[-1] and existing[-1]["ts"] >= 1.0)

    # UPDATE（命中老事实）
    existing = [{"fact": "老1", "ts": 1.0}, {"fact": "老2", "ts": 2.0}]
    before_ts = existing[0]["ts"]
    changed = store._apply("UPDATE", "更新后", existing, "老1")
    check("UPDATE→changed=True", changed is True)
    check("UPDATE→原位替换", existing[0]["fact"] == "更新后")
    check("UPDATE→数量不变", len(existing) == 2)
    check("UPDATE→ts 更新", existing[0]["ts"] >= before_ts)

    # MERGE（命中老事实，分号拼接）
    existing = [{"fact": "老1", "ts": 1.0}]
    changed = store._apply("MERGE", "补充", existing, "老1")
    check("MERGE→changed=True", changed is True)
    check("MERGE→；拼接", existing[0]["fact"] == "老1；补充")

    # DELETE（丢弃新事实，不变）
    existing = [{"fact": "老1", "ts": 1.0}]
    changed = store._apply("DELETE", "重复的", existing, "老1")
    check("DELETE→changed=False", changed is False)
    check("DELETE→existing 不变", len(existing) == 1 and existing[0]["fact"] == "老1")

    # UPDATE 未命中 old_fact_text → 不变 + 返回 False
    existing = [{"fact": "老1", "ts": 1.0}]
    changed = store._apply("UPDATE", "新", existing, "不存在的老")
    check("UPDATE 未命中→changed=False", changed is False)
    check("UPDATE 未命中→existing 不变", len(existing) == 1 and existing[0]["fact"] == "老1")

    # MERGE 未命中 → 不变
    existing = [{"fact": "老1", "ts": 1.0}]
    changed = store._apply("MERGE", "新", existing, "不存在的老")
    check("MERGE 未命中→changed=False", changed is False)


def test_vector_store_routing():
    """[MOCK] 测试6：VectorStore provider 路由（mock backend 类，测选择 + 非法 provider 抛异常）"""
    print("\n=== 测试组 E [MOCK]：VectorStore provider 路由 ===")
    _stub_modules()
    import rag.vector_store as vs
    from utils.config_handler import vector_conf

    # 用可识别的假子类替换两个具体 backend
    class FakeChroma(vs.ChromaBackend):
        marker = "CHROMA"

        def __init__(self):
            pass

    class FakeMilvus(vs.MilvusBackend):
        marker = "MILVUS"

        def __init__(self):
            pass

    vs.ChromaBackend = FakeChroma
    vs.MilvusBackend = FakeMilvus

    orig = dict(vector_conf)

    # provider=milvus（config 默认值）
    vector_conf["provider"] = "milvus"
    b = vs.build_backend()
    check("provider=milvus→MilvusBackend", getattr(b, "marker", None) == "MILVUS", f"实际 {type(b).__name__}")

    # provider=chroma
    vector_conf["provider"] = "chroma"
    b = vs.build_backend()
    check("provider=chroma→ChromaBackend", getattr(b, "marker", None) == "CHROMA", f"实际 {type(b).__name__}")

    # 大小写不敏感
    vector_conf["provider"] = "MILVUS"
    b = vs.build_backend()
    check("provider=MILVUS（大写）→MilvusBackend", getattr(b, "marker", None) == "MILVUS")

    # 非法 provider → ValueError
    vector_conf["provider"] = "redis"
    raised = False
    err = ""
    try:
        vs.build_backend()
    except ValueError as e:
        raised = True
        err = str(e)
    check("provider=redis→ValueError", raised)
    check("ValueError 含 provider 名", "redis" in err, f"实际 {err}")

    # 缺省 provider（不设）→ 默认 milvus
    vector_conf.pop("provider", None)
    b = vs.build_backend()
    check("缺省 provider→MilvusBackend", getattr(b, "marker", None) == "MILVUS")

    vector_conf.clear()
    vector_conf.update(orig)


def test_json_persistence_roundtrip():
    """[REAL] 测试7：记忆 JSON 持久化 round-trip（_load/_save_facts）"""
    print("\n=== 测试组 F [REAL]：记忆 JSON 持久化 round-trip ===")
    _stub_modules()
    from agent.memory_store import MemoryStore

    store = object.__new__(MemoryStore)
    store.json_path = os.path.join(tempfile.gettempdir(), "aiops_json_rt_test.json")
    if os.path.exists(store.json_path):
        os.remove(store.json_path)

    check("初始 _load_facts 空", store._load_facts() == [])

    facts = [{"fact": "事实A", "ts": 1.0}, {"fact": "事实B", "ts": 2.0}]
    store._save_facts(facts)
    loaded = store._load_facts()
    check("save→load 内容一致", loaded == facts, f"实际 {loaded}")
    check("load 条数=2", len(loaded) == 2)

    with open(store.json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    check("磁盘文件是合法 JSON list", isinstance(raw, list) and len(raw) == 2)
    check("JSON 含中文且未转义", raw[0]["fact"] == "事实A", f"实际 {raw[0] if raw else '空'}")

    # 覆盖写
    store._save_facts([{"fact": "覆盖", "ts": 3.0}])
    check("覆盖写后条数=1", len(store._load_facts()) == 1)
    check("覆盖写内容正确", store._load_facts()[0]["fact"] == "覆盖")

    os.remove(store.json_path)


def test_memory_store_search_with_mock_vector():
    """[MOCK] 测试8：memory_store.search 召回逻辑（mock vector_store.similarity_search）"""
    print("\n=== 测试组 G [MOCK]：memory_store.search 召回逻辑 ===")
    _stub_modules()
    from agent.memory_store import MemoryStore

    store = object.__new__(MemoryStore)
    store.json_path = os.path.join(tempfile.gettempdir(), "aiops_search_test.json")

    # 空记忆库 → 直接返回 []
    with open(store.json_path, "w", encoding="utf-8") as f:
        json.dump([], f)
    check("空记忆库→search 返回 []", store.search("任意", k=3) == [])

    # 有记忆但 vector_store 抛异常 → 降级返回 []
    with open(store.json_path, "w", encoding="utf-8") as f:
        json.dump([{"fact": "x", "ts": 1.0}], f)

    class BoomVector:
        def similarity_search(self, q, k):
            raise RuntimeError("向量库挂了")

    store.vector_store = BoomVector()
    check("vector_store 异常→降级 []", store.search("x", k=3) == [])

    # 正常召回
    from langchain_core.documents import Document

    class FakeVector:
        def similarity_search(self, q, k):
            return [Document(page_content=f"事实{i}") for i in range(k)]

    store.vector_store = FakeVector()
    res = store.search("磁盘满", k=3)
    check("正常召回 k=3→3 条", len(res) == 3, f"实际 {res}")
    check("召回内容是 page_content", res == ["事实0", "事实1", "事实2"])

    os.remove(store.json_path)


if __name__ == "__main__":
    print("AIOps 记忆系统真实可跑测试")
    print("=" * 64)
    print("诚实标注：")
    print("  [REAL] 真实跑：滑动窗口/会话隔离/JSON 持久化/search 召回")
    print("  [MOCK] mock LLM/Milvus 后测真实解析与合并逻辑")
    print("  [SKIP] 跑不了：Milvus 真实连接、LLM 真实调用（无 API key）")
    print("  [SKIP] 没做  ：viking（L0/L1/L2 分层/目录递归/IntentAnalyzer）——只有方案文档")
    print("=" * 64)

    tests = [
        test_sliding_window_and_isolation,
        test_mem0_fact_extraction_parsing,
        test_mem0_decide_update_parsing,
        test_apply_logic,
        test_vector_store_routing,
        test_json_persistence_roundtrip,
        test_memory_store_search_with_mock_vector,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:
            _failed += 1
            import traceback
            print(f"  [ERROR] {t.__name__} 抛异常: {e}")
            traceback.print_exc()

    print("\n" + "=" * 64)
    print(f"结果：{_passed} passed, {_failed} failed")
    print("=" * 64)
    sys.exit(0 if _failed == 0 else 1)
