"""
记忆检索评测（Roadmap：引入 LoCoMo / LongMemEval 思路）。

为什么要有这个脚本：
    记忆系统的核心指标不是"能不能存进去"，而是"该 recall 的时候 recall 不 recall得回来"。
    viking 相对 mem0 的核心卖点之一是"召回失败可诊断"（trace 可视化），而诊断的前提是
    先把指标量化。本脚本用可控的合成语料离线跑评测，不需要 API key。

语料构造参考 LoCoMo / LongMemEval 的关键思路：
    - 多会话（session）× 多轮对话，且刻意混入大量闲聊与不相关话题（噪声）
    - gold 问答对：问"某次故障怎么解决的"，gold 是当初沉淀的那条记忆
    - 干扰项：语义相近但不同的故障（防止靠关键词蒙对）

评测对象对比：
    - 基线：扁平 top-k（mem0 式 / 传统 RAG）——直接对所有条目 L0 做向量 top-k
    - 实验：viking 目录递归检索——先扫目录级 L0 定位目录，再目录内扫条目级 L0，L1 默认终点

指标：
    - Recall@K      gold 记忆是否出现在前 K 条召回里
    - Precision@K   召回里 gold 的占比
    - MRR           第一个 gold 出现位置的倒数排名
    - 检索步数      分层检索的路径长度（trace 步数，可诊断性间接指标）

口径说明（避免自欺）：
    - gold 按 (session, 主题) 去重。同一主题可能沉淀多条记忆，不去重会让"完美命中"
      只按重复计数打折，指标凭空少一半。
    - 无 gold 的闲聊查询（期望"什么都不召回"）只计入 Recall/Precision 的噪声惩罚，
      不进 MRR 均值 —— 没有 relevant doc 时 MRR 无定义，硬算 0 会拉低所有基线。
    - 语料用的是确定性假 embedding，绝对分数只作回归基线，不能当真实效果读。

用法：
    python tests/eval_memory_retrieval.py                 # 跑评测并打印报告
    python tests/eval_memory_retrieval.py --k 5           # 指定召回条数
    python tests/eval_memory_retrieval.py --out report.md # 额外写出报告文件
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.viking.directory_retrieval import DirectoryRecursiveRetriever  # noqa: E402
from agent.viking.intent_analyzer import (  # noqa: E402
    ContextType,
    IntentAnalyzer,
    TypedQuery,
)
from agent.viking.l0_index import InMemoryL0Index, L0Index  # noqa: E402
from agent.viking.viking_fs import MemoryEntry, VikingFS  # noqa: E402

# ---------------- 确定性 embedding（离线可跑） ----------------

def _stable_hash(s: str) -> int:
    """稳定哈希：内置 hash() 带随机化，会让评测结果不可复现。"""
    import hashlib
    return int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:8], "big")


def hash_embed(text, dim=128):
    v = [0.0] * dim
    for i, ch in enumerate(text):
        v[_stable_hash(ch) % dim] += 1.0
        if i + 1 < len(text):
            v[(_stable_hash(text[i:i + 2]) + i) % dim] += 1.0
    return v


# ---------------- 合成语料 ----------------

# 每条：(session_id, 分类, 事实原文, l0, l1, 是否 gold 主题)
CORPUS = [
    ("s1", "incidents",
     "order-service 磁盘满，根因是日志未配置 logrotate 轮转，磁盘写满导致写入失败",
     "order-service 磁盘满：日志未轮转",
     "根因是日志未配置 logrotate 轮转，已加 crontab 定时任务清理旧日志", "disk"),
    ("s1", "solutions",
     "给 order-service 配置 logrotate，保留 7 天日志",
     "order-service 日志轮转方案",
     "用 logrotate 按天轮转，保留 7 天，避免磁盘写满", "disk"),
    ("s2", "incidents",
     "payment-service CPU 飙高，根因是 JVM 堆内存配置过小导致频繁 full gc",
     "payment-service CPU 飙高：full gc 频繁",
     "根因是 JVM 堆内存配置过小，已调大 -Xmx 并加 gc 监控", "cpu"),
    ("s2", "preferences",
     "用户要求运维结论用中文、分点输出，并附影响面说明",
     "用户偏好中文分点输出",
     "用户偏好结构化中文回复，需附影响面", "pref"),
    ("s3", "incidents",
     "gateway-service 502 增多，根因是上游 user-service 连接池耗尽",
     "gateway 502 增多：上游连接池耗尽",
     "根因是 user-service 连接池耗尽，已扩容连接池并加熔断", "net"),
    ("s3", "decisions",
     "决定把 user-service 连接池从 50 调到 200，理由是高峰期耗尽",
     "user-service 连接池扩容决策",
     "连接池 50 → 200，理由是解决高峰期耗尽", "net"),
    ("s4", "user_profile",
     "用户负责 order-service 与 payment-service，日常做值班巡检",
     "用户负责 order/payment 服务巡检",
     "用户是 order/payment 服务的值班负责人", "prof"),
    ("s4", "misc",
     "今天团建，下午三点放假",
     "团建通知",
     "团建活动通知", None),
]

# 噪声（不产生记忆但出现在对话里，用于稀释 gold）
NOISE = [
    "今天天气不错", "午饭吃了什么", "这个需求什么时候上线",
    "帮我看看周报模板", "新的门禁卡在哪领", "会议室几点关门",
    "内网是不是又抽风了", "周五下午有技术分享",
]

QUERIES = [
    ("order-service 磁盘满了怎么解决", ["disk"]),
    ("日志轮转该怎么配置", ["disk"]),
    ("payment-service CPU 飙高是什么原因", ["cpu"]),
    ("用户希望我怎么回复运维结论", ["pref"]),
    ("gateway 502 增多的根因是什么", ["net"]),
    ("user-service 连接池为什么要扩容", ["net"]),
    ("用户负责哪些服务", ["prof"]),
    ("今天下午有什么安排", []),                    # 无 gold：纯闲聊，期望召回为空
]


def build_vfs(tmp: str) -> VikingFS:
    vfs = VikingFS(base_dir=os.path.join(tmp, "eval_fs"))
    for sid, cat, l2, l0, l1, theme in CORPUS:
        vfs.ensure_category(cat)
        entry = MemoryEntry(
            entry_id=f"{sid}-{theme or 'noise'}",
            category=cat, l0=l0, l1=l1, l2=l2, session_id=sid,
        )
        vfs.write(entry, ensure_dir=False)
    # 目录级 L0：模拟 LLM 为各目录生成的 .abstract.md
    vfs.write_dir_meta("incidents", "本目录：历史故障记录，含磁盘/CPU/网络类故障",
                       "按故障类型组织，条目为根因定位")
    vfs.write_dir_meta("solutions", "本目录：验证过的解决方案", "按方案类型组织")
    vfs.write_dir_meta("preferences", "本目录：用户交互偏好", "按偏好类型组织")
    vfs.write_dir_meta("decisions", "本目录：历史决策与理由", "按决策主题组织")
    vfs.write_dir_meta("user_profile", "本目录：用户画像", "按画像维度组织")
    return vfs


# ---------------- 检索策略 ----------------

class RuleBasedAnalyzer:
    """离线的确定性 intent analyzer：把 query 原样当一个 MEMORY TypedQuery 交给检索。

    为什么不用真 LLM：本评测要量的是"同一份输入下，扁平 top-k vs 目录递归"的差值。
    一旦用真 LLM 做意图分析，两组实验的输入就不可控了（意图不同、步数 25+、结果不可复现），
    测出来的差异分不清是检索结构带来的还是意图分析带来的。
    """

    def analyze(self, query, session_summary="", last_messages=None) -> list[TypedQuery]:
        return [TypedQuery(query=query, context_type=ContextType.MEMORY,
                           intent="rule", priority=3)]


def flat_baseline(index: L0Index, query: str, k: int) -> list[str]:
    """基线 mem0 式扁平 top-k：不做目录定位，直接对所有条目 L0 取 top-k。"""
    hits = index.search(query, k=k * 2, level="entry")
    return [h[0].metadata.get("entry_id", "") for h in hits[:k]]


def viking_retrieve(ret: DirectoryRecursiveRetriever, query: str, k: int) -> tuple[list[str], int]:
    """viking 目录递归检索，返回 (entry_ids, trace 步数)。"""
    result = ret.search(query, k=k)
    ids = [it.entry.entry_id for it in result.items]
    return ids, len(result.trace.steps)


# ---------------- 评测 ----------------

def evaluate(k: int, verbose: bool = True, use_llm_intent: bool = False) -> dict:
    tmp = tempfile.mkdtemp(prefix="viking_eval_")
    try:
        vfs = build_vfs(tmp)
        index = InMemoryL0Index(embedder=hash_embed)
        index.upsert([index_doc(vfs, e) for e in vfs.list_entries()])
        index.upsert(dir_docs(vfs))

        analyzer = IntentAnalyzer() if use_llm_intent else RuleBasedAnalyzer()
        ret = DirectoryRecursiveRetriever(
            vfs=vfs, index=index, intent_analyzer=analyzer,
            dir_threshold=0.12, entry_threshold=0.12, detail_depth=2
        )

        stats = {
            "flat": {"recall": [], "precision": [], "mrr": [], "hit_layer": []},
            "viking": {"recall": [], "precision": [], "mrr": [], "hops": []},
        }
        rows = []
        for query, gold_themes in QUERIES:
            # 同 (session, 主题) 可能沉淀多条记忆 → 去重，否则完美命中会被重复计数打折
            gold_ids = sorted({
                f"{sid}-{t}" for sid, _cat, _l2, _l0, _l1, t in CORPUS if t in gold_themes
            })

            flat_ids = flat_baseline(index, query, k)
            vid, hops = viking_retrieve(ret, query, k)

            r_f = recall_at(flat_ids, gold_ids, k)
            p_f = precision_at(flat_ids, gold_ids)
            r_v = recall_at(vid, gold_ids, k)
            p_v = precision_at(vid, gold_ids)
            # 无 gold（期望静默）时 MRR 无定义：不进均值，只记噪声召回
            mrr_f = mrr(flat_ids, gold_ids) if gold_ids else None
            mrr_v = mrr(vid, gold_ids) if gold_ids else None

            for side, (r, p, m) in {"flat": (r_f, p_f, mrr_f), "viking": (r_v, p_v, mrr_v)}.items():
                stats[side]["recall"].append(r)
                stats[side]["precision"].append(p)
                stats[side]["mrr"].append(m)
            stats["viking"]["hops"].append(hops)
            rows.append({
                "query": query, "gold": gold_ids, "flat": flat_ids, "viking": vid,
                "r_f": r_f, "r_v": r_v, "hops": hops,
                "m_f": mrr_f, "m_v": mrr_v,
            })

        if verbose:
            print_report(rows, stats, k)
        return {"rows": rows, "stats": stats, "k": k}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def index_doc(vfs: VikingFS, e: MemoryEntry):
    from langchain_core.documents import Document
    return Document(
        page_content=e.l0,
        metadata={"level": "entry", "category": e.category,
                  "entry_id": e.entry_id, "path": e.path},
    )


def dir_docs(vfs: VikingFS):
    from langchain_core.documents import Document
    docs = []
    for cat in vfs.list_categories():
        ab, _ = vfs.read_dir_meta(cat)
        if ab:
            docs.append(Document(
                page_content=ab,
                metadata={"level": "dir", "category": cat,
                          "path": f"memories/{cat}", "entry_id": ""},
            ))
    return docs


def recall_at(ids: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return 1.0 if not ids else 0.0     # 无 gold 却召回 = 噪声
    hit = len(set(ids) & set(gold))
    return hit / len(gold)


def precision_at(ids: list[str], gold: list[str]) -> float:
    if not ids:
        return 1.0 if not gold else 0.0
    return len(set(ids) & set(gold)) / len(ids)


def mrr(ids: list[str], gold: list[str]) -> float:
    for i, x in enumerate(ids, start=1):
        if x in gold:
            return 1.0 / i
    return 0.0


def avg(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def print_report(rows, stats, k):
    print("=" * 78)
    print(f"记忆检索评测：扁平 top-k（mem0 式基线） vs viking 目录递归检索  @K={k}")
    print("=" * 78)
    print(f"{'查询':<32}{'gold':<14}{'扁平R@K':<10}{'分层R@K':<10}{'步数':<6}")
    print("-" * 78)
    for r in rows:
        g = ",".join(x.split("-", 1)[1] for x in r["gold"]) or "-"
        mark = "" if r["m_v"] is not None else "  (无 gold，不进 MRR)"
        print(f"{r['query'][:30]:<32}{g:<14}{r['r_f']:<10.2f}{r['r_v']:<10.2f}"
              f"{r['hops']:<6}{mark}")
    print("-" * 78)
    for side, name in (("flat", "基线·扁平 top-k"), ("viking", "实验·viking 分层")):
        s = stats[side]
        print(f"{name:<16} Recall@{k}={avg(s['recall']):.3f}  "
              f"Precision={avg(s['precision']):.3f}  MRR={avg([x for x in s['mrr'] if x is not None]):.3f}")
    print(f"{'检索步数':<16} 平均 {avg(stats['viking']['hops']):.1f} 步（trace 可诊断性）")
    print("-" * 78)
    print("注：R@K 中无 gold 的查询若被召回则记 0（噪声惩罚）；MRR 不含无 gold 的查询。")
    print("注：语料用确定性假 embedding，绝对分数只作回归基线，不代表真实模型下的效果。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--intent", action="store_true",
                    help="用真 LLM 做意图分析（默认离线路由，结果可复现；--intent 仅供对照观察）")
    args = ap.parse_args()
    result = evaluate(k=args.k, use_llm_intent=args.intent)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("# 记忆检索评测报告\n\n")
            f.write(f"对比：扁平 top-k（mem0 基线） vs viking 目录递归检索，@K={args.k}\n\n")
            f.write("| 查询 | gold | 扁平 R@K | 分层 R@K | 检索步数 |\n")
            f.write("|---|---|---|---|---|\n")
            for r in result["rows"]:
                g = ",".join(r["gold"]) or "-"
                f.write(f"| {r['query']} | {g} | {r['r_f']:.2f} | {r['r_v']:.2f} | {r['hops']} |\n")
            for side, name in (("flat", "基线·扁平 top-k"), ("viking", "实验·viking 分层")):
                s = result["stats"][side]
                f.write(f"\n**{name}**：Recall@{args.k}={avg(s['recall']):.3f}, "
                        f"Precision={avg(s['precision']):.3f}, MRR={avg(s['mrr']):.3f}\n")
        print(f"\n报告已写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
