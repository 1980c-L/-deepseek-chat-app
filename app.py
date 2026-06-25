"""
DeepSeek AI Chat — Streamlit App
简洁专业的 AI 聊天界面，基于 DeepSeek API
"""
import streamlit as st
import openai
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# ── 加载环境变量 ───────────────────────────────────────────
load_dotenv()

# ── 页面配置 ───────────────────────────────────────────────
st.set_page_config(
    page_title="DeepSeek Chat",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── 样式 ───────────────────────────────────────────────────
st.markdown(
    """
<style>
    /* 隐藏默认页脚 */
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}

    /* 聊天气泡 */
    .stChatMessage {
        border-radius: 12px;
        padding: 8px 4px;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
        line-height: 1.7;
    }

    /* 输入框优化 */
    [data-testid="stChatInput"] textarea {
        border-radius: 10px !important;
        border: 1px solid #e0e0e0 !important;
        padding: 12px 16px !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #4F46E5 !important;
        box-shadow: 0 0 0 2px rgba(79,70,229,0.1) !important;
    }

    /* 标题区 */
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

# ── 侧边栏 ─────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://api.deepseek.com/_next/image?url=%2Flogo.png&w=96&q=75",
        width=40,
    )
    st.markdown("## ⚙️ 设置")

    # API Key
    api_key = st.text_input(
        "🔑 DeepSeek API Key",
        type="password",
        value=os.getenv("DEEPSEEK_API_KEY", ""),
        placeholder="sk-...",
        help="在 https://platform.deepseek.com/api_keys 获取",
    )

    st.divider()

    # 模型选择
    model = st.selectbox(
        "🧠 模型",
        options=["deepseek-chat", "deepseek-reasoner"],
        index=0,
        help="deepseek-chat: 通用对话 | deepseek-reasoner: 深度推理",
    )

    # 温度
    temperature = st.slider(
        "🌡️ 温度",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="越高越随机，越低越确定",
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

    # 清空按钮
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # 关于
    with st.expander("ℹ️ 关于"):
        st.markdown(
            """
        **DeepSeek Chat**  
        基于 Streamlit + DeepSeek API 构建  
        简洁 · 专业 · 高效
            
        [DeepSeek 官网](https://deepseek.com)  
        [API 文档](https://api-docs.deepseek.com)
        """
        )

# ── 标题 ───────────────────────────────────────────────────
st.markdown(
    '<p class="app-title">🤖 DeepSeek Chat</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="app-subtitle">由 DeepSeek API 驱动的智能对话助手</p>',
    unsafe_allow_html=True,
)

# ── 初始化会话状态 ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── 渲染历史消息 ───────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=msg.get("avatar")):
        st.markdown(msg["content"])

# ── 用户输入 ───────────────────────────────────────────────
if prompt := st.chat_input("输入你的问题，按 Enter 发送…"):
    # 校验 API Key
    if not api_key:
        st.error("❌ 请在侧边栏填写 DeepSeek API Key")
        st.stop()

    # 添加用户消息
    st.session_state.messages.append(
        {"role": "user", "content": prompt, "avatar": "👤"}
    )

    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # 调用 DeepSeek API（流式）
    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        full_response = ""

        try:
            client = openai.OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com",
            )

            stream = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一个专业、友好的 AI 助手。"},
                    *[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                ],
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

    # 保存助手回复
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response, "avatar": "🤖"}
    )
