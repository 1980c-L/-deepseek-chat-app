"""
GLM AI Chat — Streamlit App
简洁专业的 AI 聊天界面，基于智谱 GLM API（支持视觉理解）
"""
import streamlit as st
import openai
import os
import json
import base64
import time
import re
from io import BytesIO
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
from rag import DocumentStore
from agent_tools import get_langchain_tools

# ── 加载环境变量 ───────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("ZHIPU_API_KEY", "")

# ── 页面配置 ───────────────────────────────────────────────
st.set_page_config(
    page_title="GLM Chat",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── 样式 ───────────────────────────────────────────────────
st.markdown(
    """
<style>
    /* ═══════════════════════════════════════════════════ */
    /*  全局基础                                        */
    /* ═══════════════════════════════════════════════════ */
    footer, #MainMenu { visibility: hidden; }

    /* 页面背景渐变 */
    .stApp {
        background: linear-gradient(160deg, #080810 0%, #0d0d20 40%, #101028 100%) !important;
    }

    /* 内容区限宽居中 */
    .main .block-container {
        max-width: 900px !important;
        padding: 1.5rem 24px 0 24px !important;
    }

    /* 滚动条 */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #2a2a45; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #3d3d60; }

    /* ═══════════════════════════════════════════════════ */
    /*  标题                                            */
    /* ═══════════════════════════════════════════════════ */
    .app-title {
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #a78bfa 0%, #6366f1 50%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: -0.3rem;
    }
    .app-subtitle {
        font-size: 0.85rem;
        color: #6b6b80;
        margin-bottom: 0.5rem;
    }

    /* ═══════════════════════════════════════════════════ */
    /*  侧边栏                                          */
    /* ═══════════════════════════════════════════════════ */
    [data-testid="stSidebar"] {
        background: linear-gradient(175deg, #0a0a18, #0f0f24) !important;
        border-right: 1px solid #1c1c35 !important;
    }
    .sidebar-section {
        color: #a78bfa !important;
        font-size: 0.7rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-top: 18px;
        margin-bottom: 6px;
        opacity: 0.85;
    }
    /* 侧边栏按钮 */
    [data-testid="stSidebar"] .stButton button {
        border-radius: 8px !important;
        font-size: 0.82rem !important;
        transition: all 0.2s ease;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(99,102,241,0.15);
    }
    /* 侧边栏分割线 */
    [data-testid="stSidebar"] hr {
        border-color: #1c1c35 !important;
        margin: 14px 0 !important;
    }

    /* ═══════════════════════════════════════════════════ */
    /*  聊天气泡                                        */
    /* ═══════════════════════════════════════════════════ */
    [data-testid="stChatMessage"] {
        border-radius: 16px !important;
        padding: 14px 18px !important;
        margin-bottom: 14px !important;
        background: #14142b !important;
        border: 1px solid #1e1e3a !important;
        box-shadow: 0 2px 12px rgba(0,0,0,0.15);
        animation: msgFadeIn 0.4s ease-out;
    }
    @keyframes msgFadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
        line-height: 1.8;
        margin-bottom: 4px;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p code {
        background: #1e1e3a;
        padding: 2px 6px;
        border-radius: 4px;
        color: #a78bfa;
        font-size: 0.87em;
    }

    /* ═══════════════════════════════════════════════════ */
    /*  输入区 + 发送按钮                                */
    /* ═══════════════════════════════════════════════════ */
    [data-testid="stChatInput"] {
        position: sticky !important;
        bottom: 0 !important;
        z-index: 50 !important;
        background: linear-gradient(0deg, #080810 70%, transparent 100%) !important;
        padding: 12px 0 16px 0 !important;
    }
    [data-testid="stChatInput"] textarea {
        border-radius: 14px !important;
        border: 1.5px solid #252545 !important;
        padding: 14px 18px !important;
        background: #12122b !important;
        color: #e8e8f0 !important;
        font-size: 0.94rem !important;
        box-shadow: 0 2px 12px rgba(0,0,0,0.25);
        transition: all 0.25s ease;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #818cf8 !important;
        box-shadow: 0 0 0 3px rgba(129,140,248,0.12), 0 4px 20px rgba(99,102,241,0.15) !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #505070 !important;
    }

    /* ═══════════════════════════════════════════════════ */
    /*  按钮（全局）                                     */
    /* ═══════════════════════════════════════════════════ */
    .stButton > button {
        border-radius: 10px !important;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }

    /* ═══════════════════════════════════════════════════ */
    /*  Toggle 开关                                      */
    /* ═══════════════════════════════════════════════════ */
    [data-testid="stToggle"] label {
        color: #c0c0d0 !important;
    }

    /* ═══════════════════════════════════════════════════ */
    /*  Typing 动画                                      */
    /* ═══════════════════════════════════════════════════ */
    .typing-dots {
        display: inline-flex; gap: 5px; align-items: center; padding: 10px 0;
    }
    .typing-dots span {
        width: 7px; height: 7px; border-radius: 50%;
        background: linear-gradient(135deg, #6366f1, #a78bfa);
        box-shadow: 0 0 8px rgba(99,102,241,0.4);
        animation: bounceDots 1.4s infinite ease-in-out;
    }
    .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
    .typing-dots span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes bounceDots {
        0%, 60%, 100% { transform: translateY(0); opacity: 0.3; }
        30% { transform: translateY(-10px); opacity: 1; }
    }

    /* ═══════════════════════════════════════════════════ */
    /*  代码块                                          */
    /* ═══════════════════════════════════════════════════ */
    [data-testid="stMarkdownContainer"] pre {
        background: #0a0a18 !important;
        border: 1px solid #202045 !important;
        border-radius: 12px !important;
        padding: 18px !important;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.4);
    }
    [data-testid="stMarkdownContainer"] code {
        font-family: 'JetBrains Mono','Fira Code','Cascadia Code',monospace !important;
        font-size: 0.85rem !important;
    }
    [data-testid="stMarkdownContainer"] pre code {
        color: #c8c8dc !important;
    }

    /* ═══════════════════════════════════════════════════ */
    /* 头像图片 */
    .avatar-img {
        border-radius: 50% !important;
        object-fit: cover;
    }
    /* 上传区 + 提示框 */
    /* ═══════════════════════════════════════════════════ */
    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed #252545 !important;
        border-radius: 12px !important;
        padding: 8px 14px !important;
        background: rgba(99,102,241,0.03) !important;
        transition: all 0.2s ease;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #818cf8 !important;
        background: rgba(99,102,241,0.06) !important;
    }
    .stAlert {
        border-radius: 12px !important;
        border: 1px solid #252545 !important;
        background: #13132e !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ── 智谱 API 配置 ─────────────────────────────────────────
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
VISION_MODEL = "GLM-4V-Flash"       # 免费视觉理解
TEXT_MODEL = "GLM-4-Flash-250414"   # 免费纯文本（备选）
HISTORY_FILE = Path(__file__).parent / ".chat_history.json"

# ── RAG 文档存储 ───────────────────────────────────────────
if "doc_store" not in st.session_state:
    st.session_state.doc_store = DocumentStore(Path(__file__).parent / ".rag_cache")
if "rag_enabled" not in st.session_state:
    st.session_state.rag_enabled = False
if "agent_enabled" not in st.session_state:
    st.session_state.agent_enabled = False
if "user_avatar" not in st.session_state:
    st.session_state.user_avatar = "👤"
if "ai_avatar" not in st.session_state:
    st.session_state.ai_avatar = "🤖"
if "user_avatar_img" not in st.session_state:
    st.session_state.user_avatar_img = None
if "ai_avatar_img" not in st.session_state:
    st.session_state.ai_avatar_img = None


# ── 对话持久化 ────────────────────────────────────────────
def save_history(messages: list):
    """保存对话到本地 JSON 文件"""
    try:
        slim = []
        for m in messages:
            entry = {"role": m["role"], "content": m["content"]}
            # 不持久化图片（太大），只记有图片的事实
            if m.get("images") and len(m["images"]) > 0:
                entry["had_images"] = len(m["images"])
            slim.append(entry)
        HISTORY_FILE.write_text(json.dumps(slim, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass  # 保存失败不阻塞正常使用


def load_history() -> list:
    """从本地 JSON 文件加载对话"""
    try:
        if HISTORY_FILE.exists():
            raw = HISTORY_FILE.read_text(encoding="utf-8")
            slim = json.loads(raw)
            messages = []
            for entry in slim:
                msg = {
                    "role": entry["role"],
                    "content": entry["content"],
                    "avatar": st.session_state.user_avatar if entry["role"] == "user" else st.session_state.ai_avatar,
                }
                if entry.get("had_images"):
                    msg["images"] = ["[历史图片]"] * entry["had_images"]
                messages.append(msg)
            return messages
    except Exception:
        pass
    return []


# ── 错误提示 ──────────────────────────────────────────────
def friendly_error(error: Exception) -> str:
    """把各种 API 异常翻译成人话"""
    msg = str(error)
    if isinstance(error, openai.AuthenticationError):
        return "🔒 API Key 无效，请检查 .env 文件中的 ZHIPU_API_KEY"
    if isinstance(error, openai.RateLimitError):
        return "⏳ 请求太频繁，请稍等一下再试"
    if isinstance(error, openai.APITimeoutError):
        return "⏱️ API 响应超时，请检查网络后重试"
    if isinstance(error, openai.APIConnectionError):
        return "🌐 网络连接失败，请检查网络 / 代理是否正常"
    if isinstance(error, openai.InternalServerError):
        return "🔧 智谱服务暂时异常，稍后重试"
    if "1210" in msg:
        return "⚙️ 参数错误（如 max_tokens 超出模型限制）"
    if "1000" in msg or "1001" in msg:
        return "🔑 API Key 未设置或格式错误"
    if "1115" in msg:
        return "💰 余额不足，请前往 open.bigmodel.cn 充值"
    if "quota" in msg.lower() or "insufficient" in msg.lower():
        return "💰 免费额度已用完或余额不足"
    return f"⚠️ 请求出错：{error}"


# ── 侧边栏 ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ 设置")

    # ── 连接 ──
    st.markdown('<p class="sidebar-section">🔌 连接</p>', unsafe_allow_html=True)
    if API_KEY:
        masked = API_KEY[:8] + "…" + API_KEY[-4:]
        st.success(f"API Key: {masked}")
    else:
        st.error("❌ 未找到 ZHIPU_API_KEY")
        st.caption("在 `.env` 文件中设置：")
        st.code("ZHIPU_API_KEY=你的key", language="bash")

    st.divider()

    # ── 模型参数 ──
    st.markdown('<p class="sidebar-section">🧠 模型参数</p>', unsafe_allow_html=True)
    model = st.selectbox(
        "模型",
        options=[VISION_MODEL, TEXT_MODEL],
        index=0,
        help=f"{VISION_MODEL}: 支持图片理解 | {TEXT_MODEL}: 纯文本",
    )
    temperature = st.slider("温度", 0.0, 2.0, 0.7, 0.1)
    max_tokens = st.slider("最大输出", 64, 1024, 512, 64)

    # Agent 模式
    agent_enabled = st.toggle("🤖 Agent 模式", value=st.session_state.agent_enabled,
                              help="开启后 AI 可调用工具：计算、搜索、读写文件")
    if agent_enabled != st.session_state.agent_enabled:
        st.session_state.agent_enabled = agent_enabled
    if agent_enabled:
        st.caption("可用工具：🔢计算器 🔍搜索 🌐抓网页 📂文件读写")

    st.divider()

    # ── 文档 RAG ──
    st.markdown('<p class="sidebar-section">📚 文档问答</p>', unsafe_allow_html=True)

    # RAG 开关
    rag_enabled = st.toggle("🔍 基于文档回答", value=st.session_state.rag_enabled,
                            help="开启后 AI 将基于上传的文档内容回答")
    if rag_enabled != st.session_state.rag_enabled:
        st.session_state.rag_enabled = rag_enabled

    # 文档上传
    if rag_enabled:
        doc_files = st.file_uploader(
            "上传文档",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
            key="doc_uploader",
            label_visibility="collapsed",
        )
        if doc_files:
            ds = st.session_state.doc_store
            new_docs = [f for f in doc_files if f.name not in ds.doc_names]
            if new_docs:
                with st.spinner("解析文档…"):
                    total = 0
                    bar = st.progress(0)
                    for i, f in enumerate(new_docs):
                        n = ds.add_document(f.name, f.read(), API_KEY, ZHIPU_BASE_URL)
                        total += n
                        bar.progress((i + 1) / len(new_docs))
                    bar.empty()
                st.success(f"已索引 {len(new_docs)} 个文档，{total} 个片段")
                st.rerun()

        # 已加载文档列表
        ds = st.session_state.doc_store
        if ds.doc_names:
            st.caption(f"📄 {len(ds.doc_names)} 个文档已加载")
            for name in ds.doc_names:
                st.caption(f"  • {name}")
            if st.button("清除文档", use_container_width=True):
                ds.clear()
                st.session_state.rag_enabled = False
                st.rerun()

    st.divider()

    # ── 对话操作 ──
    st.markdown('<p class="sidebar-section">💬 对话</p>', unsafe_allow_html=True)

    # 清空：两步确认
    if "confirm_clear" not in st.session_state:
        st.session_state.confirm_clear = False

    col1, col2 = st.columns(2)
    with col1:
        if not st.session_state.confirm_clear:
            if st.button("🗑️ 清空", use_container_width=True):
                st.session_state.confirm_clear = True
                st.rerun()
        else:
            if st.button("⚠️ 确认清空", use_container_width=True, type="primary"):
                st.session_state.messages = []
                st.session_state.uploaded_images = []
                st.session_state.confirm_clear = False
                try:
                    HISTORY_FILE.unlink(missing_ok=True)
                except Exception:
                    pass
                st.rerun()
            if st.button("↩ 取消", use_container_width=True):
                st.session_state.confirm_clear = False
                st.rerun()

    with col2:
        if st.button("📂 加载", use_container_width=True):
            with st.spinner("加载历史记录…"):
                bar = st.progress(0, text="读取中…")
                loaded = load_history()
                bar.progress(50, text="解析中…")
                time.sleep(0.3)
                bar.progress(100, text="完成")
                time.sleep(0.3)
                bar.empty()
            if loaded:
                st.session_state.messages = loaded
                st.success(f"已恢复 {len(loaded)} 条消息")
                st.rerun()
            else:
                st.info("没有历史记录")

    st.divider()

    # ── 外观 ──
    st.markdown('<p class="sidebar-section">🎨 外观</p>', unsafe_allow_html=True)

    col_ua, col_aa = st.columns(2)
    with col_ua:
        st.caption("你的头像")
        user_av = st.text_input("你", value=st.session_state.user_avatar,
                                max_chars=2, key="user_av_input",
                                label_visibility="collapsed",
                                placeholder="👤")
        if user_av.strip():
            st.session_state.user_avatar = user_av.strip()
            st.session_state.user_avatar_img = None
        user_img_file = st.file_uploader("上传", type=["png","jpg","jpeg","webp"],
                                         key="uimg", label_visibility="collapsed")
        if user_img_file:
            st.session_state.user_avatar_img = Image.open(user_img_file).convert("RGBA")

    with col_aa:
        st.caption("AI 头像")
        ai_av = st.text_input("AI", value=st.session_state.ai_avatar,
                              max_chars=2, key="ai_av_input",
                              label_visibility="collapsed",
                              placeholder="🤖")
        if ai_av.strip():
            st.session_state.ai_avatar = ai_av.strip()
            st.session_state.ai_avatar_img = None
        ai_img_file = st.file_uploader("上传", type=["png","jpg","jpeg","webp"],
                                       key="aimg", label_visibility="collapsed")
        if ai_img_file:
            st.session_state.ai_avatar_img = Image.open(ai_img_file).convert("RGBA")

    # 快速预设
    presets = st.radio(
        "快速预设",
        ["👤/🤖 默认", "😊/🧠 表情", "🐱/🐶 动物", "🧑‍💻/🤖 程序员", "🐼/🦊 萌系"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )
    if st.button("应用预设", use_container_width=True):
        mapping = {
            "👤/🤖 默认": ("👤", "🤖"),
            "😊/🧠 表情": ("😊", "🧠"),
            "🐱/🐶 动物": ("🐱", "🐶"),
            "🧑‍💻/🤖 程序员": ("🧑‍💻", "🤖"),
            "🐼/🦊 萌系": ("🐼", "🦊"),
        }
        st.session_state.user_avatar, st.session_state.ai_avatar = mapping[presets]
        st.session_state.user_avatar_img = None
        st.session_state.ai_avatar_img = None
        st.rerun()

    st.divider()

    with st.expander("📖 使用指南"):
        st.markdown(
            """
### 🚀 快速上手

| 模式 | 怎么用 | 能干嘛 |
|------|--------|--------|
| **默认聊天** | 直接打字发消息 | 日常对话、写代码、翻译 |
| **📎 传图片** | 点 📎 选图，打字提问 | 识别图中内容、OCR |
| **🌐 读网页** | 贴 URL 到输入框 | 抓取网页 + AI 摘要 |
| **🔍 文档问答** | 开 RAG 开关 → 上传 PDF | 基于文档精准回答 |
| **🤖 Agent** | 开 Agent 开关 → 打字 | AI 自动搜索、计算、读写文件 |

### 💡 试试这些

- `帮我算 (123+456)*789 等于多少`
- `搜索一下 DeepSeek 最新消息`
- `写一首关于编程的诗，保存到 code-poem.txt`
- 贴一个新闻链接，问 "这篇文章讲了什么"
- 传一张截图，问 "帮我提取上面的文字"

### ⚙️ 配置提示

- API Key 在 `.env` 文件配置，不需要每次填
- 切换模型在侧边栏 **🧠 模型参数**
- 对话自动保存，刷新不丢失
- 清空前会二次确认，不怕误删
        """
        )

# ── 标题 ───────────────────────────────────────────────────
st.markdown('<p class="app-title">🤖 GLM Chat</p>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">由智谱 GLM 驱动的多模态对话助手</p>', unsafe_allow_html=True)

# ── 初始化 + 加载历史 ──────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = load_history()
if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = []

# ── 新手欢迎 ───────────────────────────────────────────────
if not st.session_state.messages:
    st.info(
        "👋 **欢迎！** 我是基于智谱 GLM 的 AI 助手。\n\n"
        "💬 直接聊天 · 📎 上传图片识别 · 🌐 贴链接分析网页 · 🔍 文档问答 · 🤖 Agent 自动执行\n\n"
        "左下角 **📖 使用指南** 有更多示例，试试看吧！",
    )


def encode_image(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def build_messages(model_name: str, user_text: str, images: list[Image.Image]) -> list[dict]:
    # RAG 上下文注入
    rag_context = ""
    if st.session_state.rag_enabled:
        ds = st.session_state.doc_store
        if ds.is_ready and API_KEY:
            rag_context = ds.query(user_text, API_KEY, ZHIPU_BASE_URL)

    if rag_context:
        system_prompt = (
            "你是一个专业的文档问答助手。请严格基于以下文档片段回答用户问题。"
            "如果文档中没有相关信息，请明确说'文档中未提及'，不要编造。\n\n"
            f"## 参考文档\n{rag_context}"
        )
    else:
        system_prompt = "你是一个专业、友好的 AI 助手，支持理解用户上传的图片。"

    system_msg = {"role": "system", "content": system_prompt}
    history = []
    for m in st.session_state.messages:
        # 跳过历史图片占位符
        if isinstance(m["content"], str):
            history.append({"role": m["role"], "content": m["content"]})

    if images and VISION_MODEL in model_name:
        content = [{"type": "text", "text": user_text}]
        for img in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": encode_image(img)},
            })
        current = {"role": "user", "content": content}
    else:
        current = {"role": "user", "content": user_text}

    return [system_msg] + history + [current]


def get_avatar_tuple(which: str) -> tuple:
    """返回 (emoji_str_or_None, image_or_None)"""
    if which == "user":
        return (st.session_state.user_avatar, st.session_state.user_avatar_img)
    else:
        return (st.session_state.ai_avatar, st.session_state.ai_avatar_img)


# ── 渲染历史消息 ───────────────────────────────────────────
for msg in st.session_state.messages:
    role = msg["role"]
    emoji, img = get_avatar_tuple(role)
    # 有图片头像时不用 chat_message 内置 avatar，自己画
    if img is not None:
        c_av, c_msg = st.columns([0.07, 0.93], gap="small")
        with c_av:
            st.image(img, width=36)
        with c_msg:
            content = msg["content"]
            if isinstance(content, str):
                st.markdown(content)
            if msg.get("images"):
                for im in msg["images"]:
                    if isinstance(im, Image.Image):
                        st.image(im, width=200)
    else:
        with st.chat_message(role, avatar=emoji):
            content = msg["content"]
            if isinstance(content, str):
                st.markdown(content)
            if msg.get("images"):
                for im in msg["images"]:
                    if isinstance(im, Image.Image):
                        st.image(im, width=200)

# ── 底部输入栏 ─────────────────────────────────────────────
st.markdown(
    """
<style>
    div[data-testid="stFileUploader"] { margin-bottom: 4px; }
    div[data-testid="stFileUploader"] button {
        font-size: 0.8rem !important;
        padding: 2px 10px !important;
        min-height: 26px !important;
        border-radius: 6px !important;
    }
    div[data-testid="stFileUploader"] section { padding: 0 !important; }
</style>
""",
    unsafe_allow_html=True,
)

c_url, c_img = st.columns([0.7, 0.3], gap="small")

with c_url:
    browse_url = st.text_input(
        "🌐",
        placeholder="粘贴 URL 让 AI 分析网页…",
        key="url_input",
        label_visibility="collapsed",
    )

with c_img:
    uploaded_files = st.file_uploader(
        "📎 选择图片",
        type=["png", "jpg", "jpeg", "webp", "gif"],
        accept_multiple_files=True,
        key="image_uploader",
        label_visibility="collapsed",
    )

current_images = []
if uploaded_files:
    for f in uploaded_files:
        img = Image.open(f).convert("RGB")
        current_images.append(img)

prompt = st.chat_input("输入你的问题，按 Enter 发送…")

# 如果有 URL 且没输 prompt，自动生成
if browse_url and not prompt:
    prompt = f"请分析这个网页的内容：{browse_url}"

if prompt:
    if not API_KEY:
        st.error("❌ 未检测到 API Key！请在项目 `.env` 文件中设置 `ZHIPU_API_KEY=你的key`")
        st.stop()

    st.session_state.uploaded_images = current_images

    with st.chat_message("user", avatar=st.session_state.user_avatar if not st.session_state.user_avatar_img else None):
        st.markdown(prompt)
        for img in current_images:
            st.image(img, width=200)

    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "avatar": st.session_state.user_avatar,
        "images": [img.copy() for img in current_images] if current_images else [],
    })

    with st.chat_message("assistant", avatar=st.session_state.ai_avatar):
        placeholder = st.empty()
        full_response = ""
        error_occurred = False

        # 显示 typing 动画
        typing_html = (
            '<div class="typing-dots">'
            "<span></span><span></span><span></span>"
            "</div>"
        )
        placeholder.markdown(typing_html, unsafe_allow_html=True)

        try:
            client = openai.OpenAI(
                api_key=API_KEY,
                base_url=ZHIPU_BASE_URL,
                timeout=30.0,
            )

            actual_model = VISION_MODEL if current_images else model

            if st.session_state.agent_enabled:
                # ── Agent 模式：LangChain 工具调用 ──
                from langchain_openai import ChatOpenAI
                from langgraph.prebuilt import create_react_agent

                # RAG 上下文（Agent 模式也支持）
                agent_rag_context = ""
                if st.session_state.rag_enabled:
                    ds = st.session_state.doc_store
                    if ds.is_ready and API_KEY:
                        agent_rag_context = ds.query(prompt, API_KEY, ZHIPU_BASE_URL)

                llm = ChatOpenAI(
                    model=actual_model,
                    api_key=API_KEY,
                    base_url=ZHIPU_BASE_URL,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                tools = get_langchain_tools()
                agent = create_react_agent(llm, tools)

                # 构建历史（只取文本消息）
                agent_messages = []
                for m in st.session_state.messages[-10:]:  # 最近 10 条
                    if isinstance(m.get("content"), str):
                        agent_messages.append(m["content"])

                # 显示思考中
                placeholder.markdown("🤔 思考中…")

                # 流式执行 Agent
                final_output = ""
                tool_logs = []
                for event in agent.stream(
                    {"messages": [{"role": "user", "content": f"## 参考文档\n{agent_rag_context}\n\n## 用户问题\n{prompt}" if agent_rag_context else prompt,
                                   "context": "\n".join(agent_messages[-6:])}]},
                    stream_mode="values",
                ):
                    if "messages" in event:
                        msgs = event["messages"]
                        last = msgs[-1] if msgs else None
                        if last:
                            msg_type = getattr(last, "type", "")
                            if msg_type == "tool":
                                tool_name = getattr(last, "name", "?")
                                tool_logs.append(f"🔧 调用 **{tool_name}**")
                                placeholder.markdown("\n".join(tool_logs))
                            elif msg_type == "ai":
                                ai_content = getattr(last, "content", "")
                                if ai_content:
                                    final_output = ai_content
                                    placeholder.markdown(
                                        "\n".join(tool_logs + ["", final_output])
                                    )

                if not final_output.strip():
                    final_output = "Agent 未产生有效回复，请简化问题或关闭 Agent 模式重试。"

                # 如果有工具调用，最终回复带工具日志
                if tool_logs:
                    full_response = "\n".join(tool_logs + ["", "---", "", final_output])
                else:
                    full_response = final_output

            else:
                # ── 普通模式：直接对话 ──
                messages = build_messages(actual_model, prompt, current_images)
                stream = client.chat.completions.create(
                    model=actual_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )

                for chunk in stream:
                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", None) or ""
                    reasoning = getattr(delta, "reasoning_content", None)
                    if content or reasoning:
                        if content:
                            full_response += content
                        else:
                            full_response += reasoning
                        placeholder.markdown(full_response)

                # 兜底
                if not full_response.strip():
                    full_response = chunk.choices[0].message.content if hasattr(chunk.choices[0], "message") else ""

                # 异常检测
                text_only = re.sub(r'<[^>]+>', '', full_response).strip()
                if full_response.strip() and not text_only:
                    full_response = "⚠️ API 返回了异常响应，请稍后重试或切换模型。"
                elif not full_response.strip():
                    full_response = "⚠️ 未收到有效回复，请重试。"

            placeholder.markdown(full_response)
            save_history(st.session_state.messages + [{
                "role": "assistant", "content": full_response, "avatar": st.session_state.ai_avatar,
            }])

        except Exception as e:
            error_occurred = True
            st.error(friendly_error(e))
            st.session_state.messages.pop()

    if not error_occurred:
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "avatar": st.session_state.ai_avatar,
        })
        save_history(st.session_state.messages)

    st.session_state.uploaded_images = []
    st.rerun()
