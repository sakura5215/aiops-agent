"""
viking IntentAnalyzer：把用户 query 拆成 0-5 个 TypedQuery，决定去哪类上下文里检索。

对应方案文档 §4.1（viking 文档的机制）与 §4.6（find/search 的分流，工程判断）：

- viking 的 IntentAnalyzer 用 LLM 分析意图，输入是 [会话压缩摘要 + 最近 5 条消息 + 当前 query]，
  输出 0-5 个 TypedQuery(query / context_type / intent / priority)。
- 不同 context_type 用不同的 query 改写风格：skill 动词优先、resource 名词短语、
  memory 用 "User's XX"。目的是让改写后的 query 主动适配目标存储的语义空间。
- 0 个 TypedQuery = 闲聊/问候，不需要检索，直接走 ReAct。
- find() 不做 intent analysis（简单查询，省一次 LLM 调用），search() 才做。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

from utils.logger_handler import logger

MAX_TYPED_QUERIES = 5


class ContextType(str, Enum):
    """viking 三类上下文根目录。AIOps 适配：记忆 / 知识库 / 运维工具。"""

    MEMORY = "MEMORY"      # → memories/  用户记忆与历史故障
    RESOURCE = "RESOURCE"  # → resources/ 运维知识库文档
    SKILL = "SKILL"        # → skills/    运维工具调用经验


@dataclass
class TypedQuery:
    """IntentAnalyzer 的输出单元。priority 1-5，用于多路递归检索的优先级队列。"""

    query: str
    context_type: ContextType
    intent: str = ""
    priority: int = 3

    @property
    def root(self) -> str:
        """viking Root Directory Mapping：MEMORY→memories / RESOURCE→resources / SKILL→skills"""
        return {
            ContextType.MEMORY: "memories",
            ContextType.RESOURCE: "resources",
            ContextType.SKILL: "skills",
        }[ContextType(self.context_type)]


# 不同 context_type 的 query 改写风格（viking 文档明确）
QUERY_STYLE = {
    ContextType.MEMORY: "写成 \"User's XX\" 形式，例如 \"User's code style preferences\"",
    ContextType.RESOURCE: "写成名词短语，例如 \"API usage guide\"",
    ContextType.SKILL: "动词开头，例如 \"Check disk usage metric\"",
}

PROMPT = """你是查询意图分析器。给定当前用户输入与近期会话上下文，判断需要检索哪几类上下文，
并为每一类生成一条检索查询。

可用的上下文类型：
- MEMORY：用户记忆、历史故障、历史决策（虚拟路径 memories/）
- RESOURCE：运维知识库文档（虚拟路径 resources/）
- SKILL：运维工具调用经验（虚拟路径 skills/）

不同上下文类型的查询要按不同风格改写：
{memory_style}
{resource_style}
{skill_style}

判定规则：
- 闲聊、问候、纯寒暄（如"你好""今天天气不错"）不检索，输出 NONE。
- 一次输入可能同时需要多类上下文（例如既问历史故障、又问知识库文档、又要用工具），此时输出多行。
- 每行严格四列，用竖线分隔：
  `上下文类型 | 改写后的查询 | 查询目的（一句话） | 优先级(1-5)`
- 优先级按对回答的重要程度给，5 最高。最多 {max_n} 行。

近期会话上下文：
{context}

当前用户输入：
{query}

输出（多行，每行列四段；无则输出 NONE）："""


def parse_typed_queries(content: str, default_type: ContextType = ContextType.MEMORY) -> list[TypedQuery]:
    """解析 LLM 输出的 TypedQuery 列表（纯函数，不调 LLM，便于单测）。

    容错：识别 `MEMORY | query | intent | 5` 四列形式，也接受只有两列
    `MEMORY | query` 的退化形式。非法行丢弃，不抛异常。
    """
    out: list[TypedQuery] = []
    if not content:
        return out
    if content.strip().upper().startswith("NONE"):
        return out
    for raw in content.splitlines():
        line = raw.strip()
        if not line or line.upper().startswith("NONE"):
            continue
        # 去掉可能的前缀编号，如 "1. MEMORY | ..."
        line = re.sub(r"^\s*\d+[\.、)]\s*", "", line)
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue
        raw_type = parts[0].strip().upper()
        try:
            ctx = ContextType(raw_type)
        except ValueError:
            ctx = default_type
        query = parts[1]
        if not query:
            continue
        intent = parts[2] if len(parts) > 2 else ""
        priority_raw = parts[3] if len(parts) > 3 else "3"
        m = re.search(r"\d+", priority_raw)
        priority = int(m.group()) if m else 3
        priority = max(1, min(5, priority))
        out.append(TypedQuery(query=query, context_type=ctx, intent=intent, priority=priority))
        if len(out) >= MAX_TYPED_QUERIES:
            break
    return out


# 中文疑问词：口语里大量问题不带问号（"磁盘满怎么解决"），只判 ? / 吗 / 呢 会误判成
# 简单查询，把本该走多意图检索的问题降级成 find()，漏掉 RESOURCE/SKILL 侧的意图。
QUESTION_WORDS = ("怎么", "怎样", "如何", "为什么", "为何", "哪些", "哪个", "什么",
                  "多少", "能否", "能不能", "可不可以", "是不是")


def should_use_search(query: str) -> bool:
    """find() 还是 search()（方案文档 §4.6 的工程判断，非 viking 既定机制）。

    AIOps 的运维问题天然多意图，默认走 search()；只把"短且无疑问"的输入降级到 find()，
    省掉一次 IntentAnalyzer 的 LLM 调用。

    疑问判定同时看：问号、句末语气词、中文疑问词。因为中文口语提问很少打问号。
    """
    q = (query or "").strip()
    if not q:
        return False
    if "?" in q or "？" in q:
        return True
    if q.endswith(("吗", "呢")):
        return True
    if any(w in q for w in QUESTION_WORDS):
        return True
    if len(q) >= 10:
        return True
    return False


class IntentAnalyzer:
    """viking IntentAnalyzer 的 AIOps 实现：产出 0-5 个 TypedQuery。"""

    def __init__(self, model=None):
        from model.factory import chat_model
        self.model = model or chat_model

    def build_prompt(self, query: str, session_summary: str = "",
                     last_messages: Optional[Sequence[str]] = None) -> str:
        ctx_parts = []
        if session_summary:
            ctx_parts.append(f"[会话压缩摘要] {session_summary}")
        if last_messages:
            ctx_parts.append("[最近对话] " + " / ".join(str(m) for m in list(last_messages)[-5:]))
        return PROMPT.format(
            memory_style=QUERY_STYLE[ContextType.MEMORY],
            resource_style=QUERY_STYLE[ContextType.RESOURCE],
            skill_style=QUERY_STYLE[ContextType.SKILL],
            max_n=MAX_TYPED_QUERIES,
            context="\n".join(ctx_parts) or "（无）",
            query=query,
        )

    def analyze(self, query: str, session_summary: str = "",
                last_messages: Optional[Sequence[str]] = None) -> list[TypedQuery]:
        """完整意图分析：调 LLM 得到 0-5 个 TypedQuery。LLM 失败时降级为单条 MEMORY 查询。"""
        try:
            resp = self.model.invoke(
                self.build_prompt(query, session_summary, last_messages)
            )
            content = resp.content if isinstance(resp.content, str) else str(resp.content)
            return parse_typed_queries(content)
        except Exception as e:
            logger.warning(f"[intent_analyzer]意图分析失败，降级为单条记忆查询: {e}")
            return [TypedQuery(query=query, context_type=ContextType.MEMORY,
                               intent="fallback", priority=3)]

    def plan(self, query: str, session_summary: str = "",
             last_messages: Optional[Sequence[str]] = None) -> tuple[bool, list[TypedQuery]]:
        """决定走 search() 还是 find()，返回 (use_search, typed_queries)。"""
        if not should_use_search(query):
            return False, [TypedQuery(query=query, context_type=ContextType.MEMORY,
                                      intent="simple", priority=3)]
        return True, self.analyze(query, session_summary, last_messages)
