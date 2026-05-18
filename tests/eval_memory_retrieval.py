"""
记忆检索评测（Roadmap：引入 LoCoMo / LongMemEval 思路）。

为什么要有这个脚本：
    记忆系统的核心指标不是"能不能存进去"，而是"该 recall 的时候 recall 不 recall得回来"。
    viking 相对 mem0 的核心卖点之一是"召回失败可诊断"（trace 可视化），而诊断的前提是
    先把指标量化。本脚本用从公开真实生产故障复盘提炼的语料离线跑评测，不需要 API key。

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
    - 语料用确定性假 embedding，绝对分数只作回归基线，不能当真实效果读；
      但语料内容提炼自公开真实故障复盘（非自造），检索命题与生产场景同分布。

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


# ---------------- 真实语料（提炼自公开生产故障复盘） ----------------
# 语料不再自造，而是从公开真实生产故障复盘文章提炼，每条记忆标注出处，
# 覆盖 CPU 飙高 / 内存 OOM / 磁盘满 / 连接池耗尽 / 慢 SQL 五类高发故障。
# 出处：阿里云开发者社区、达梦社区、BestHub、椰云网络、网硕互联、精创网络、北冥有鱼 等公开技术复盘。
# theme 用 c1~c9 唯一标识一个具体故障案例（每个案例 = 1 条 incident + 1 条 solution），
# 使 gold 精确到"具体某次故障"，而非粗粒度故障类别，避免 gold 集合被过度放大。

# 每条：(session_id, 分类, 事实原文, l0, l1, 是否 gold 主题)
CORPUS = [
    # --- c1：定时任务叠加导致 CPU 持续 100%（椰云网络）---
    ("s1", "incidents",
     "电商服务器 CPU 持续 100%，根因是日志处理定时任务每分钟执行一次、单次执行时间超过间隔导致任务无限叠加",
     "CPU 持续 100%：定时任务叠加",
     "根因是定时任务执行时间超过间隔导致叠加，已优化逻辑并加锁防重叠", "c1"),
    ("s1", "solutions",
     "给日志处理定时任务加执行锁，并调整执行间隔使其大于单次执行时间",
     "定时任务叠加的治理方案",
     "定时任务加锁防重叠，间隔大于单次执行时间", "c1"),

    # --- c2：Full GC + 备份锁副本导致 CPU 飙高（BestHub）---
    ("s2", "incidents",
     "order-service 凌晨 CPU 98%，根因是 MySQL 从库被备份任务锁住导致查询超时、线程阻塞、Full GC 频繁",
     "order-service CPU 飙高：备份锁从库",
     "根因是备份锁住从库导致查询超时线程阻塞，已杀备份任务并加熔断", "c2"),
    ("s2", "solutions",
     "把备份移到低峰期并加 --single-transaction 无锁备份，JVM 堆从 4G 提到 8G，加从库复制延迟监控",
     "备份锁从库的治理方案",
     "备份移低峰期 + 无锁备份 + 堆扩容 + 复制延迟监控", "c2"),

    # --- c3：Druid SQL 缓存泄漏 + SQL 拼接导致 OOM（阿里云开发者社区）---
    ("s3", "incidents",
     "线上服务 OOM 崩溃，根因是 Druid 1.1.22 SQL 统计缓存无限制缓存 SQL 字符串 + 业务循环拼接超大 INSERT SQL",
     "服务 OOM：Druid SQL 缓存泄漏",
     "根因是 Druid 1.1.22 SQL 缓存泄漏叠加业务 SQL 拼接，已升级并改批量插入", "c3"),
    ("s3", "solutions",
     "升级 Druid 修复 SQL 缓存泄漏，业务改批量插入避免循环拼接超大 SQL",
     "Druid OOM 的治理方案",
     "升级 Druid + 关闭 SQL 统计缓存 + 改批量插入", "c3"),

    # --- c4：磁盘 I/O 瓶颈连锁引发数据库 OOM（达梦社区）---
    ("s4", "incidents",
     "数据库主备切换后 8000 连接耗尽内存触发 OOM，根因是磁盘 I/O 性能不足导致 SQL 阻塞、连接雪崩式增长",
     "数据库 OOM：磁盘 I/O 瓶颈",
     "根因是磁盘 I/O 不足导致 SQL 阻塞连接积压，已换 SSD 并扩容内存", "c4"),
    ("s4", "solutions",
     "更换高性能 SSD 保证写入吞吐 150MB/s 以上，内存扩容到 256GB，切断 I/O 瓶颈到 OOM 的故障链",
     "磁盘 I/O 导致 OOM 的治理",
     "换 SSD + 扩容内存 + 切断 I/O→SQL超时→连接积压→OOM 链路", "c4"),

    # --- c5：慢 SQL 无索引导致连接池耗尽三连宕机（BestHub）---
    ("s5", "incidents",
     "后端整体卡死三次，根因是二维码登录轮询 SQL 的 scene 字段缺索引导致全表扫描、连接池耗尽",
     "后端卡死：慢 SQL 打满连接池",
     "根因是登录轮询 SQL 缺索引全表扫，已给 scene 字段加索引", "c5"),
    ("s5", "solutions",
     "给 scene 字段加索引让查询走索引，EXPLAIN 确认后响应恢复、连接池占用回落",
     "慢 SQL 打满连接池的治理",
     "给高频轮询 SQL 加索引，连接池先扩容止血再找慢 SQL", "c5"),

    # --- c6：create_time 缺索引午间活动打满连接池（建站侠）---
    ("s6", "incidents",
     "每天中午 12 点连接池耗尽报错，根因是订单汇总 SQL 的 create_time 缺索引全表扫，被误配到用户活动页",
     "周期性连接池耗尽：create_time 缺索引",
     "根因是 create_time 缺索引 + 管理 API 误配到活动页，已加索引并路由隔离", "c6"),
    ("s6", "solutions",
     "给 create_time 加索引，管理功能 API 与用户 API 路由隔离，批量查询用独立小连接池",
     "周期性连接池耗尽的治理",
     "加索引 + 管理 API 隔离 + 批量查询独立连接池", "c6"),

    # --- c7：日志未轮转导致磁盘满服务连锁崩溃（网硕互联）---
    ("s7", "incidents",
     "服务器磁盘写满导致服务连锁崩溃，根因是日志长期未轮转且无磁盘监控告警",
     "磁盘满：日志未轮转",
     "根因是日志未轮转 + 无磁盘告警，已清理日志并加 logrotate", "c7"),
    ("s7", "solutions",
     "紧急清理日志释放空间，配置 logrotate 强制轮转，磁盘使用率 70% 告警，日志与数据库分盘",
     "磁盘满的治理方案",
     "logrotate 轮转 + 磁盘 70% 告警 + 日志数据库分盘", "c7"),

    # --- c8：logrotate copytruncate 引发 I/O 风暴（精创网络）---
    ("s8", "incidents",
     "凌晨数据库节点 I/O 延迟飙升 50 倍导致白屏，根因是 logrotate 用 copytruncate 模式对 12GB 日志截断引发 I/O 风暴",
     "I/O 风暴：logrotate copytruncate",
     "根因是 copytruncate 截断大日志引发 I/O 风暴，已改 create 模式 + 信号重载", "c8"),
    ("s8", "solutions",
     "logrotate 放弃 copytruncate，改 create 模式配 USR1 信号重载，加 delaycompress 避免 I/O 风暴",
     "logrotate I/O 风暴的治理",
     "copytruncate 改 create + USR1 信号重载 + delaycompress", "c8"),

    # --- c9：Nginx stream 自循环导致日志指数增长（北冥有鱼）---
    ("s9", "incidents",
     "站点完全不可用，根因是 Nginx stream 模块 proxy_pass 指向自身形成自循环、日志一天生成 530GB",
     "站点崩溃：Nginx stream 自循环",
     "根因是 stream proxy_pass 指向自身形成自循环，已修正配置", "c9"),
    ("s9", "solutions",
     "修正 Nginx stream 配置避免 proxy_pass 指向自身，清理日志用 truncate 而非 rm（进程持有句柄空间不释放）",
     "Nginx 日志指数增长的治理",
     "修正 stream 自循环 + truncate 清日志 + logrotate 水位告警", "c9"),

    # --- 无 gold 的闲聊/画像（噪声 + 干扰）---
    ("s10", "user_profile",
     "用户负责订单服务与支付服务，日常做值班巡检",
     "用户负责订单/支付服务巡检",
     "用户是订单/支付服务的值班负责人", "prof"),
    ("s10", "misc",
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
    # c1：定时任务叠加 CPU 100%
    ("服务器 CPU 持续 100% 怎么排查", ["c1"]),
    ("定时任务叠加导致 CPU 飙高怎么治理", ["c1"]),
    # c2：Full GC + 备份锁从库
    ("凌晨 CPU 飙高和数据库备份有什么关系", ["c2"]),
    ("备份锁住从库导致 Full GC 怎么处理", ["c2"]),
    # c3：Druid OOM
    ("线上服务 OOM 崩溃怎么排查", ["c3"]),
    ("Druid SQL 缓存泄漏怎么解决", ["c3"]),
    # c4：磁盘 I/O 导致数据库 OOM
    ("数据库主备切换后 OOM 的根因是什么", ["c4"]),
    ("磁盘 I/O 瓶颈导致数据库 OOM 怎么治理", ["c4"]),
    # c5：慢 SQL 打满连接池
    ("后端反复卡死三次是什么原因", ["c5"]),
    ("登录轮询慢 SQL 打满连接池怎么解决", ["c5"]),
    # c6：create_time 缺索引
    ("每天中午连接池耗尽报错怎么排查", ["c6"]),
    ("create_time 缺索引导致连接池打满怎么办", ["c6"]),
    # c7：日志未轮转磁盘满
    ("服务器磁盘满了怎么处理", ["c7"]),
    ("日志没轮转导致磁盘满怎么治理", ["c7"]),
    # c8：logrotate copytruncate I/O 风暴
    ("logrotate 引发 I/O 风暴是什么原因", ["c8"]),
    ("copytruncate 模式有什么问题", ["c8"]),
    # c9：Nginx stream 自循环
    ("Nginx 日志一天涨到 530GB 是什么原因", ["c9"]),
    ("Nginx stream 自循环怎么解决", ["c9"]),
    # 画像 + 闲聊
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
    vfs.write_dir_meta("incidents", "本目录：历史生产故障记录，含 CPU 飙高/内存 OOM/磁盘满/连接池耗尽类故障",
                       "按故障类型组织，条目为现象与根因定位")
    vfs.write_dir_meta("solutions", "本目录：验证过的治理方案", "按方案类型组织")
    vfs.write_dir_meta("user_profile", "本目录：用户画像", "按画像维度组织")
    vfs.write_dir_meta("misc", "本目录：与故障无关的闲聊与杂项", "按主题组织")
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
