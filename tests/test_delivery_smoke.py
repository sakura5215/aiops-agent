"""
交付验收测试。

与 test_viking / test_memory_system 的区别：那两个是组件级单测（内部逻辑对不对），
这个文件站在"别人 clone 下来能不能跑、仓库干不干净"的角度做交付级验收。

  S1 交付卫生    不需要凭据：gitignore 覆盖、无硬编码密钥、全量编译、配置/语料齐全、依赖可导入
  S2 离线端到端  不需要 key：短期记忆滑动窗口与会话隔离、viking 写入→召回闭环
  S3 凭据与连通  需要 DASHSCOPE_API_KEY：真 embedding、知识库检索、9 个工具可加载（缺 key 整段跳过）
  S4 回归门禁    子进程跑两个单测脚本，断言 exit 0

用法：python tests/test_delivery_smoke.py
"""

import compileall
import hashlib
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PASS, FAIL, SKIP = 0, 0, 0

KEY_RE = re.compile(r"sk-[A-Za-z0-9]{16,}")

DEPS = {
    "langchain": "langchain",
    "langchain-core": "langchain_core",
    "langchain-community": "langchain_community",
    "langchain-chroma": "langchain_chroma",
    "langchain-milvus": "langchain_milvus",
    "langchain-text-splitters": "langchain_text_splitters",
    "langgraph": "langgraph",
    "pymilvus": "pymilvus",
    "milvus-lite": "milvus_lite",
    "chromadb": "chromadb",
    "streamlit": "streamlit",
    "dashscope": "dashscope",
    "pyyaml": "yaml",
    "pypdf": "pypdf",
    "python-dotenv": "dotenv",
}


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


def skip(name, why):
    global SKIP
    SKIP += 1
    print(f"  SKIP {name} —— {why}")


def section(title):
    print(f"\n=== {title} ===")


def _tracked_files():
    r = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.split()


# ============ S1 交付卫生 ============

def test_hygiene():
    section("S1 交付卫生（无需凭据）")

    with open(os.path.join(ROOT, ".gitignore"), encoding="utf-8") as f:
        gi = f.read()
    must = [".venv/", ".idea/", "chat_histories/", "logs/", "md5.text",
            ".env", "/milvus_data/", "/memory_fs/"]
    missing = [m for m in must if m not in gi]
    check("gitignore 覆盖运行产物与密钥文件", not missing, f"缺失: {missing}")

    leaked = []
    for rel in _tracked_files():
        try:
            with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
                text = f.read()
        except (UnicodeDecodeError, OSError, IsADirectoryError):
            continue
        for hit in KEY_RE.findall(text):
            placeholder = rel == ".env.example" and set(hit[3:]) == {"x"}
            if not placeholder:
                leaked.append((rel, hit))
    check("仓库内无真实密钥", not leaked, str(leaked))

    files = set(_tracked_files())
    check(".env.example 作为模板入库", ".env.example" in files)
    check(".env 未进版本库", ".env" not in files)

    ok = all(compileall.compile_dir(os.path.join(ROOT, d), quiet=2)
             for d in ("agent", "rag", "utils", "model", "tests"))
    ok = ok and bool(compileall.compile_file(os.path.join(ROOT, "app.py"), quiet=2))
    check("全部源码编译通过", ok)

    need = ["config/agent.yml", "config/prompt.yml", "config/rag.yml", "config/vector_store.yml",
            "prompts/main_prompt.txt", "prompts/rag_summarize.txt", "prompts/report_prompt.txt",
            "data/cpu_high_usage.txt", "data/disk_high_usage.txt", "data/memory_high_usage.txt",
            "data/service_unavailable.txt", "data/slow_response.txt"]
    missing = [f for f in need if not os.path.exists(os.path.join(ROOT, f))]
    check("配置 / 提示词 / 知识库语料齐全", not missing, f"缺失: {missing}")

    missing = [name for name, mod in DEPS.items() if importlib.util.find_spec(mod) is None]
    check("requirements 依赖全部可导入", not missing, f"未安装: {missing}")

    env = os.environ.copy()
    env.pop("DASHSCOPE_API_KEY", None)
    r = subprocess.run([sys.executable, "-c", "import model.factory"], cwd=ROOT,
                       env=env, capture_output=True, text=True)
    out = r.stdout + r.stderr
    check("无凭据时可读报错并退出",
          r.returncode != 0 and "DASHSCOPE_API_KEY" in out, out[:200])


# ============ S2 离线端到端 ============

def _hash_embed(text, dim=96):
    v = [0.0] * dim
    for i, ch in enumerate(text):
        v[int.from_bytes(hashlib.md5(ch.encode()).digest()[:8], "big") % dim] += 1.0
        if i + 1 < len(text):
            pair = text[i:i + 2].encode()
            v[(int.from_bytes(hashlib.md5(pair).digest()[:8], "big") + i) % dim] += 1.0
    return v


def test_offline_e2e(tmp):
    section("S2 离线端到端闭环（无需 API key）")

    from langchain_core.messages import HumanMessage, AIMessage
    from agent.memory import FileChatMessageHistory

    hist_dir = os.path.join(tmp, "hist")
    store = FileChatMessageHistory(session_id="smoke-a", storage_dir=hist_dir)
    for i in range(6):
        store.add_message(HumanMessage(content=f"问{i}"))
        store.add_message(AIMessage(content=f"答{i}"))
    recent = store.recent_messages(4)
    check("滑动窗口只取最近 4 条",
          len(recent) == 4 and recent[-1].content == "答5", str([m.content for m in recent]))

    other = FileChatMessageHistory(session_id="smoke-b", storage_dir=hist_dir)
    check("会话隔离：新会话读不到别人的历史", len(other.messages) == 0)

    reloaded = FileChatMessageHistory(session_id="smoke-a", storage_dir=hist_dir)
    check("历史落盘后可回读", len(reloaded.messages) == 12, str(len(reloaded.messages)))

    from agent.viking.l0_index import InMemoryL0Index
    from agent.viking.memory_viking import VikingMemoryStore

    class FakeLLM:
        def invoke(self, prompt):
            if "归入最合适" in prompt:
                return SimpleNamespace(content="incidents")
            if "生成两层" in prompt:
                return SimpleNamespace(content="L0: 下单链路慢\nL1: 根因是库存查询延迟")
            return SimpleNamespace(content="")

    fact = "order-service 下单链路变慢，根因是 inventory-service 库存查询延迟"

    def new_store():
        return VikingMemoryStore(
            base_dir=os.path.join(tmp, "fs"),
            index=InMemoryL0Index(embedder=_hash_embed),
            model=FakeLLM(),
            dir_threshold=0.05,
            entry_threshold=0.05,
        )

    store = new_store()
    entry = store.commit_fact(fact, session_id="smoke")
    check("viking 写入落库", entry is not None and entry.category == "incidents",
          str(entry))
    check("条目落盘为文件",
          bool(entry) and os.path.exists(
              os.path.join(tmp, "fs", "memories", "incidents", f"{entry.entry_id}.json")))
    check("重复写入被 hash 去重", store.commit_fact(fact, session_id="smoke") is None)

    recalled = new_store().recall("下单链路为什么慢", k=3)
    check("写入后能被召回", any("inventory-service" in t for t in recalled), str(recalled))


# ============ S3 凭据与连通性 ============

def test_connectivity():
    section("S3 凭据与连通性（需要 DASHSCOPE_API_KEY）")
    if not os.getenv("DASHSCOPE_API_KEY", "").strip():
        skip("真实模型连通性", "未设置 DASHSCOPE_API_KEY")
        return

    from model.factory import chat_model, embed_model

    vec = embed_model.embed_query("连通性自检")
    check("DashScope embedding 可调用", bool(vec) and len(vec) > 0, str(type(vec)))
    resp = chat_model.invoke("只回复两个字：可用")
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    check("Qwen 对话可调用", len(text.strip()) > 0, text[:60])

    from rag.vector_store import VectorStoreService
    docs = VectorStoreService().get_retriever().invoke("磁盘使用率过高如何排查")
    if docs:
        check("知识库检索命中", len(docs) > 0, f"{len(docs)} 条")
    else:
        skip("知识库检索命中", "知识库为空，先跑 python -m rag.vector_store 建库")

    import agent.tools.agent_tools as tools
    names = ["rag_summarize", "get_target_service", "get_time_range", "fetch_alert_data",
             "fetch_metric_data", "fetch_log_summary", "fetch_service_topology",
             "fetch_report_data", "fill_context_for_report"]
    missing = [n for n in names if not hasattr(tools, n)]
    check("9 个工具全部可加载", not missing, f"缺失: {missing}")
    alert = tools.fetch_alert_data.invoke({"service_name": "payment-service", "time_range": "今天"})
    check("mock 数据工具可调用且返回非空", bool(alert), alert[:60])


# ============ S4 回归门禁 ============

def test_regression_gate():
    section("S4 单元测试回归门禁")
    for script in ("tests/test_viking.py", "tests/test_memory_system.py"):
        r = subprocess.run([sys.executable, script], cwd=ROOT, capture_output=True, text=True)
        tail = [line for line in r.stdout.strip().splitlines() if "总计" in line or "结果" in line]
        check(f"{os.path.basename(script)} 全绿",
              r.returncode == 0, (tail[-1] if tail else r.stdout[-200:]))


def main():
    tmp = tempfile.mkdtemp(prefix="delivery_smoke_")
    try:
        test_hygiene()
        test_offline_e2e(tmp)
        test_connectivity()
        test_regression_gate()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n=== 总计: {PASS} passed, {FAIL} failed, {SKIP} skipped ===")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
