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

    .stChatMessage {
        border-radius: 12px;
        padding: 8px 4px;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
        line-height: 1.7;
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
</style>
""",
    unsafe_allow_html=True,
)

# ── 智谱 API 配置 ─────────────────────────────────────────
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
VISION_MODEL = "GLM-4V-Flash"       # 免费视觉理解
TEXT_MODEL = "GLM-4-Flash-250414"   # 免费纯文本（备选）
HISTORY_FILE = Path(__file__).parent / ".chat_history.json"


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

    # API Key 状态（只读，从环境变量读取）
    if API_KEY:
        masked = API_KEY[:8] + "…" + API_KEY[-4:]
        st.success(f"🔑 API Key: {masked}")
    else:
        st.error("❌ 未找到 ZHIPU_API_KEY")
        st.caption("请在项目目录的 `.env` 文件中设置：")
        st.code("ZHIPU_API_KEY=你的key", language="bash")

    st.divider()

    # 模型选择
    model = st.selectbox(
        "🧠 模型",
        options=[VISION_MODEL, TEXT_MODEL],
        index=0,
        help=f"{VISION_MODEL}: 支持图片理解 | {TEXT_MODEL}: 纯文本",
    )

    temperature = st.slider("🌡️ 温度", 0.0, 2.0, 0.7, 0.1)
    max_tokens = st.slider("📏 最大输出", 64, 1024, 512, 64)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.messages = []
            st.session_state.uploaded_images = []
            # 也清空磁盘上的历史
            try:
                HISTORY_FILE.unlink(missing_ok=True)
            except Exception:
                pass
            st.rerun()
    with col2:
        if st.button("📂 加载记录", use_container_width=True):
            loaded = load_history()
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
    system_msg = {"role": "system", "content": "你是一个专业、友好的 AI 助手，支持理解用户上传的图片。"}
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
                # GLM API 可能把内容放在不同字段
                content = getattr(delta, "content", None) or ""
                reasoning = getattr(delta, "reasoning_content", None)
                if content:
                    full_response += content
                    placeholder.markdown(full_response + "▌")
                elif reasoning:
                    full_response += reasoning
                    placeholder.markdown(full_response + "▌")

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
