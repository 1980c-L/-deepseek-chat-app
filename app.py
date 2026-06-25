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
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}

    /* ── 全局限宽居中 ── */
    .main .block-container {
        max-width: 900px !important;
        padding-left: 24px;
        padding-right: 24px;
    }

    /* ── 对话卡片 ── */
    [data-testid="stChatMessage"] {
        border-radius: 14px !important;
        padding: 12px 16px !important;
        margin-bottom: 12px !important;
        background: #1a1a2e !important;
        border: 1px solid #2a2a3e !important;
    }

    /* ── 行间距 ── */
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
        line-height: 1.8;
        margin-bottom: 6px;
    }

    /* ── 输入框 ── */
    [data-testid="stChatInput"] {
        position: sticky !important;
        bottom: 0 !important;
        z-index: 50 !important;
        background: #0f0f1a !important;
        padding-top: 8px !important;
        padding-bottom: 12px !important;
    }
    [data-testid="stChatInput"] textarea {
        border-radius: 10px !important;
        border: 1px solid #3a3a4a !important;
        padding: 12px 16px !important;
        background: #1a1a2e !important;
        color: #e4e4e7 !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 2px rgba(99,102,241,0.2) !important;
    }

    /* ── 标题 ── */
    .app-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #e4e4e7;
        margin-bottom: -0.5rem;
    }
    .app-subtitle {
        font-size: 0.85rem;
        color: #888;
        margin-bottom: 0.5rem;
    }

    /* ── 侧边栏分组标题 ── */
    .sidebar-section {
        color: #6366f1 !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 16px;
        margin-bottom: 8px;
    }

    /* ── Typing 动画 ── */
    .typing-dots {
        display: inline-flex;
        gap: 4px;
        align-items: center;
        padding: 8px 0;
    }
    .typing-dots span {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #6366f1;
        animation: bounce 1.2s infinite ease-in-out;
    }
    .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
    .typing-dots span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes bounce {
        0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
        30% { transform: translateY(-8px); opacity: 1; }
    }

    /* ── 代码块高亮 ── */
    [data-testid="stMarkdownContainer"] pre {
        background: #0f0f1a !important;
        border: 1px solid #2a2a3e !important;
        border-radius: 10px !important;
        padding: 16px !important;
    }
    [data-testid="stMarkdownContainer"] code {
        font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace !important;
        font-size: 0.88rem !important;
    }
    [data-testid="stMarkdownContainer"] pre code {
        color: #e4e4e7 !important;
    }

    /* ── 拖拽上传区 ── */
    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed #3a3a4a !important;
        border-radius: 10px !important;
        padding: 6px 12px !important;
        transition: border-color 0.2s;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #6366f1 !important;
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
                    "avatar": "👤" if entry["role"] == "user" else "🤖",
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

    with st.expander("ℹ️ 关于"):
        st.markdown(
            """
        **GLM Chat**
        基于 Streamlit + 智谱 GLM API
        支持图片理解 · 流式输出 · 对话持久化

        [智谱开放平台](https://open.bigmodel.cn)
        [API 文档](https://docs.bigmodel.cn)
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


# ── 渲染历史消息 ───────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=msg.get("avatar")):
        content = msg["content"]
        if isinstance(content, str):
            st.markdown(content)
        if msg.get("images"):
            for img in msg["images"]:
                if isinstance(img, Image.Image):
                    st.image(img, width=200)

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

if prompt:
    if not API_KEY:
        st.error("❌ 未检测到 API Key！请在项目 `.env` 文件中设置 `ZHIPU_API_KEY=你的key`")
        st.stop()

    st.session_state.uploaded_images = current_images

    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
        for img in current_images:
            st.image(img, width=200)

    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "avatar": "👤",
        "images": [img.copy() for img in current_images] if current_images else [],
    })

    with st.chat_message("assistant", avatar="🤖"):
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
                timeout=30.0,  # 连接 + 读取总超时
            )

            actual_model = VISION_MODEL if current_images else model
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

            # 兜底：如果流式没拿到内容，尝试非流式结果
            if not full_response.strip():
                full_response = chunk.choices[0].message.content if hasattr(chunk.choices[0], "message") else ""

            # 检测异常回复（全是标签没有实际内容）
            text_only = re.sub(r'<[^>]+>', '', full_response).strip()
            if full_response.strip() and not text_only:
                full_response = "⚠️ API 返回了异常响应，请稍后重试或切换模型。"
            elif not full_response.strip():
                full_response = "⚠️ 未收到有效回复，请重试。"

            placeholder.markdown(full_response)
            save_history(st.session_state.messages + [{
                "role": "assistant", "content": full_response, "avatar": "🤖",
            }])

        except Exception as e:
            error_occurred = True
            st.error(friendly_error(e))
            st.session_state.messages.pop()

    if not error_occurred:
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "avatar": "🤖",
        })
        save_history(st.session_state.messages)

    st.session_state.uploaded_images = []
    st.rerun()
