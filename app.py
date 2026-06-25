"""
GLM AI Chat — Streamlit App
简洁专业的 AI 聊天界面，基于智谱 GLM API（支持视觉理解）
"""
import streamlit as st
import openai
import os
import base64
from io import BytesIO
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv

# ── 加载环境变量 ───────────────────────────────────────────
load_dotenv()

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
        border: 1px solid #e0e0e0 !important;
        padding: 12px 16px !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #4F46E5 !important;
        box-shadow: 0 0 0 2px rgba(79,70,229,0.1) !important;
    }

    .app-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #1e1e1e;
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
VISION_MODEL = "GLM-4.1V-Thinking"   # 带视觉理解
TEXT_MODEL = "GLM-4.7-Flash"         # 纯文本，免费

# ── 侧边栏 ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ 设置")

    # API Key
    api_key = st.text_input(
        "🔑 智谱 API Key",
        type="password",
        value=os.getenv("ZHIPU_API_KEY", ""),
        placeholder="your-api-key...",
        help="在 https://open.bigmodel.cn 获取",
    )

    st.divider()

    # 模型选择
    model = st.selectbox(
        "🧠 模型",
        options=[VISION_MODEL, TEXT_MODEL],
        index=0,
        help=f"{VISION_MODEL}: 支持图片理解 | {TEXT_MODEL}: 纯文本，免费",
    )

    # 温度
    temperature = st.slider(
        "🌡️ 温度",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
    )

    # 最大 Token
    max_tokens = st.slider(
        "📏 最大输出长度",
        min_value=256,
        max_value=8192,
        value=4096,
        step=256,
    )

    st.divider()

    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.uploaded_images = []
        st.rerun()

    with st.expander("ℹ️ 关于"):
        st.markdown(
            """
        **GLM Chat**  
        基于 Streamlit + 智谱 GLM API 构建  
        支持图片理解 · 流式输出
        
        [智谱开放平台](https://open.bigmodel.cn)  
        [API 文档](https://docs.bigmodel.cn)
        """
        )

# ── 标题 ───────────────────────────────────────────────────
st.markdown(
    '<p class="app-title">🤖 GLM Chat</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="app-subtitle">由智谱 GLM 驱动的多模态对话助手</p>',
    unsafe_allow_html=True,
)

# ── 初始化会话状态 ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = []


def encode_image(image: Image.Image) -> str:
    """将 PIL Image 编码为 base64 data URL"""
    buf = BytesIO()
    image.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def build_messages(model_name: str, user_text: str, images: list[Image.Image]) -> list[dict]:
    """构建 OpenAI 兼容的消息列表，支持多模态"""
    system_msg = {"role": "system", "content": "你是一个专业、友好的 AI 助手，支持理解用户上传的图片。"}

    # 历史消息
    history = []
    for m in st.session_state.messages:
        history.append({"role": m["role"], "content": m["content"]})

    # 当前用户消息
    if images and VISION_MODEL in model_name:
        # 多模态：文本 + 图片
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
        st.markdown(msg["content"])
        # 显示历史图片
        if msg.get("images"):
            for img in msg["images"]:
                st.image(img, width=200)

# ── 图片上传区 ─────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "📷 上传图片（可选）",
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

# ── 用户输入 ───────────────────────────────────────────────
if prompt := st.chat_input("输入你的问题，按 Enter 发送…"):
    if not api_key:
        st.error("❌ 请在侧边栏填写智谱 API Key")
        st.stop()

    # 保存图片引用到会话
    st.session_state.uploaded_images = current_images

    # 显示用户消息
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
        for img in current_images:
            st.image(img, width=200)

    # 保存用户消息
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "avatar": "👤",
        "images": [img.copy() for img in current_images] if current_images else [],
    })

    # 调用 API
    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        full_response = ""

        try:
            client = openai.OpenAI(
                api_key=api_key,
                base_url=ZHIPU_BASE_URL,
            )

            # 自动选用视觉模型（如果有图片）
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
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)

        except openai.AuthenticationError:
            st.error("🔒 API Key 无效，请检查后重试。")
            st.session_state.messages.pop()
            st.stop()
        except openai.RateLimitError:
            st.error("⏳ API 请求频率过高，请稍后重试。")
            st.session_state.messages.pop()
            st.stop()
        except Exception as e:
            st.error(f"⚠️ 请求出错：{e}")
            st.session_state.messages.pop()
            st.stop()

    # 保存回复
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "avatar": "🤖",
    })

    # 清空上传
    st.session_state.uploaded_images = []
    st.rerun()
