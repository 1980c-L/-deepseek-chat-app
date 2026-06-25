# DeepSeek AI Chat

简洁专业的 AI 聊天网页，基于 Streamlit + DeepSeek API。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

如果下载慢，使用清华镜像：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

### 2. 配置 API Key

**方式一：创建 `.env` 文件（推荐）**

```bash
cp .env.example .env
```

然后编辑 `.env`，填入你的 DeepSeek API Key：

```
DEEPSEEK_API_KEY=sk-your-key-here
```

> 在 https://platform.deepseek.com/api_keys 获取 API Key

**方式二：启动后在侧边栏填写**

不配置 `.env` 也可以，直接在网页左侧栏输入 API Key。

### 3. 启动应用

```bash
streamlit run app.py
```

浏览器会自动打开 `http://localhost:8501`。

## 功能

- 🤖 **DeepSeek V3 / R1 模型** — 支持通用对话 (deepseek-chat) 和深度推理 (deepseek-reasoner)
- 💬 **流式输出** — 实时显示 AI 回复，体验流畅
- 🎛️ **参数调节** — 温度、最大输出长度可调
- 🧹 **一键清空** — 随时重置对话
- 🔒 **API Key 安全** — 支持 .env 文件或侧边栏输入

## 项目结构

```
deepseek-chat-app/
├── app.py             # 主应用
├── requirements.txt   # 依赖
├── .env.example       # 环境变量模板
└── README.md
```
