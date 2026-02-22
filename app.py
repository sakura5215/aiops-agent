import time
import uuid
import streamlit as st
from agent.react_agent import ReactAgent
import traceback

# ----------------------------
# 页面配置
# ----------------------------
st.set_page_config(
    page_title="OpsPilot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# 样式：极简 ChatGPT 风格 + 隐藏 Streamlit 默认元素
# ----------------------------
st.markdown("""
<style>
    /* 整体背景 */
    .stApp {
        background: #f7f7f8;
    }

    /* 隐藏 Streamlit 默认元素 */
    [data-testid="stToolbar"] {
        display: none !important;
    }

    [data-testid="stDecoration"] {
        display: none !important;
    }

    [data-testid="stHeader"] {
        display: none !important;
    }

    #MainMenu {
        visibility: hidden !important;
    }

    footer {
        visibility: hidden !important;
    }

    header {
        visibility: hidden !important;
    }

    /* 页面主体宽度 */
    .block-container {
        max-width: 980px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    html, body, [class*="css"] {
        color: #111827;
    }

    /* 侧边栏 */
    section[data-testid="stSidebar"] {
        background: #ececf1;
        border-right: 1px solid #d9d9e3;
        min-width: 290px !important;
        max-width: 290px !important;
    }

    .sidebar-title {
        font-size: 1.08rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.9rem;
    }

    /* 主标题区 */
    .main-title {
        font-size: 1.95rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
    }

    .sub-title {
        font-size: 0.98rem;
        color: #4b5563;
        line-height: 1.7;
        margin-bottom: 1.2rem;
    }

    /* 空状态 */
    .empty-wrap {
        text-align: center;
        padding-top: 9vh;
        color: #6b7280;
    }

    .empty-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.6rem;
    }

    .empty-desc {
        font-size: 1rem;
        color: #6b7280;
        line-height: 1.7;
    }

    /* 消息区 */
    div[data-testid="stChatMessage"] {
        background: transparent;
        border: none;
        padding-top: 0.3rem;
        padding-bottom: 0.3rem;
        margin-bottom: 0.35rem;
    }

    [data-testid="chat-avatar-icon-user"] svg,
    [data-testid="chat-avatar-icon-assistant"] svg {
        width: 1.15rem;
        height: 1.15rem;
    }

    /* 输入框区域 */
    .stChatInputContainer {
        background: #f7f7f8;
    }

    /* 历史记录说明 */
    .history-tip {
        font-size: 0.85rem;
        color: #6b7280;
        line-height: 1.6;
        margin-top: 0.8rem;
    }

    .quick-label {
        font-size: 0.92rem;
        font-weight: 600;
        color: #4b5563;
        margin-bottom: 0.75rem;
    }

    hr {
        border-color: #e5e7eb;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    /* 通用按钮 */
    div.stButton > button {
        width: 100%;
        border-radius: 14px;
        border: 1px solid #d1d5db;
        background: #ffffff;
        color: #111827;
        font-weight: 500;
        padding: 0.62rem 0.85rem;
        box-shadow: none;
    }

    div.stButton > button:hover {
        background: #f3f4f6;
        border-color: #c7cdd4;
        color: #111827;
    }

    /* 左侧历史小框按钮更像卡片 */
    section[data-testid="stSidebar"] div.stButton > button {
        text-align: left !important;
        justify-content: flex-start !important;
        border-radius: 12px;
        border: 1px solid #d7d9df;
        background: #f8f8fb;
        color: #111827;
        padding: 0.72rem 0.85rem;
        margin-bottom: 0.35rem;
        font-size: 0.92rem;
        min-height: 48px;
    }

    section[data-testid="stSidebar"] div.stButton > button:hover {
        background: #ffffff;
        border-color: #bfc5cf;
    }

    /* 当前选中的历史记录卡片 */
    .active-history {
        background: #ffffff;
        border: 1px solid #bfc5cf;
        border-radius: 12px;
        padding: 0.72rem 0.85rem;
        margin-bottom: 0.35rem;
        color: #111827;
        font-size: 0.92rem;
        font-weight: 600;
    }

    /* 新建对话按钮 */
    .new-chat-wrap {
        margin-bottom: 0.8rem;
    }

    /* 常用问题按钮更宽更长 */
    .quick-question-note {
        font-size: 0.88rem;
        color: #6b7280;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ----------------------------
# 初始化
# ----------------------------
if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

if "conversations" not in st.session_state:
    first_id = str(uuid.uuid4())
    st.session_state["conversations"] = {
        first_id: {
            "title": "新对话",
            "messages": []
        }
    }
    st.session_state["current_conversation_id"] = first_id

if "current_conversation_id" not in st.session_state:
    st.session_state["current_conversation_id"] = next(iter(st.session_state["conversations"]))

if "pending_prompt" not in st.session_state:
    st.session_state["pending_prompt"] = None

def get_current_conversation():
    conv_id = st.session_state["current_conversation_id"]
    return st.session_state["conversations"][conv_id]


def build_title_from_messages(messages: list[dict]) -> str:
    for msg in messages:
        if msg["role"] == "user" and msg["content"].strip():
            title = msg["content"].strip().replace("\n", " ")
            return title[:22] + ("..." if len(title) > 22 else "")
    return "新对话"


def create_new_conversation():
    new_id = str(uuid.uuid4())
    st.session_state["conversations"][new_id] = {
        "title": "新对话",
        "messages": []
    }
    st.session_state["current_conversation_id"] = new_id
    st.session_state["pending_prompt"] = None


# ----------------------------
# 左侧历史栏
# ----------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">OpsPilot</div>', unsafe_allow_html=True)

    if st.button("＋ 新建对话", key="new_chat_btn"):
        create_new_conversation()
        st.rerun()

    st.markdown("---")

    conversation_ids = list(st.session_state["conversations"].keys())

    # 倒序显示，最近的在上面
    for cid in reversed(conversation_ids):
        title = st.session_state["conversations"][cid]["title"]
        display_title = title if title.strip() else "新对话"

        if cid == st.session_state["current_conversation_id"]:
            st.markdown(
                f'<div class="active-history">{display_title}</div>',
                unsafe_allow_html=True
            )
        else:
            if st.button(display_title, key=f"history_{cid}"):
                st.session_state["current_conversation_id"] = cid
                st.session_state["pending_prompt"] = None
                st.rerun()

    st.markdown(
        '<div class="history-tip">左侧可以访问历史会话，右侧用于提问与查看回答。</div>',
        unsafe_allow_html=True
    )


# ----------------------------
# 右侧主聊天区
# ----------------------------
current_conv = get_current_conversation()
messages = current_conv["messages"]

st.markdown(
    """
    <div class="main-title">企业 AIOps 智能体 OpsPilot</div>
    <div class="sub-title">
        支持运维知识问答、告警分析、日志与监控辅助诊断，以及运行报告生成。
    </div>
    """,
    unsafe_allow_html=True
)

if len(messages) == 0:
    st.markdown("""
    <div class="empty-wrap">
        <div class="empty-title">今天想分析什么问题？</div>
        <div class="empty-desc">
            你可以直接提问，例如：<br>
            分析一下 order-service 最近1小时的异常情况
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="quick-label">常用问题</div>', unsafe_allow_html=True)
    st.markdown('<div class="quick-question-note">点击下面的问题可直接开始对话。</div>', unsafe_allow_html=True)

    # 改成长一点：两行两列
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)

    with row1_col1:
        if st.button("分析一下 order-service 最近1小时的异常情况", key="quick_q_1"):
            st.session_state["pending_prompt"] = "分析一下 order-service 最近1小时的异常情况"
            st.rerun()

    with row1_col2:
        if st.button("payment-service 今天有什么主要告警", key="quick_q_2"):
            st.session_state["pending_prompt"] = "payment-service 今天有什么主要告警"
            st.rerun()

    with row2_col1:
        if st.button("生成 inventory-service 本月运行报告", key="quick_q_3"):
            st.session_state["pending_prompt"] = "生成 inventory-service 本月运行报告"
            st.rerun()

    with row2_col2:
        if st.button("CPU 使用率过高一般怎么排查", key="quick_q_4"):
            st.session_state["pending_prompt"] = "CPU 使用率过高一般怎么排查"
            st.rerun()

    st.markdown("---")


# ----------------------------
# 展示当前会话消息
# ----------------------------
for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ----------------------------
# 输入处理
# ----------------------------
user_input = st.chat_input("请输入你的问题...")

if st.session_state["pending_prompt"]:
    prompt = st.session_state["pending_prompt"]
    st.session_state["pending_prompt"] = None
elif user_input:
    prompt = user_input
else:
    prompt = None


# ----------------------------
# 流式输出
# ----------------------------
if prompt:
    current_conv = get_current_conversation()
    conv_id = st.session_state["current_conversation_id"]

    # 前端messages仍可保留用于界面展示，但不再用于agent
    current_conv["messages"].append({"role": "user", "content": prompt})
    current_conv["title"] = build_title_from_messages(current_conv["messages"])

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        try:
            with st.spinner("智能客服思考中..."):
                # 传入conv_id
                stream = st.session_state["agent"].execute_stream(conv_id, prompt)

                for chunk in stream:
                    if chunk:
                        full_response += chunk
                        placeholder.markdown(full_response + "▌")

                placeholder.markdown(full_response)

        except Exception as e:
            traceback.print_exc()
            full_response = f"系统运行出错：{type(e).__name__}: {str(e)}"
            placeholder.error(full_response)

    current_conv["messages"].append({"role": "assistant", "content": full_response})
    current_conv["title"] = build_title_from_messages(current_conv["messages"])