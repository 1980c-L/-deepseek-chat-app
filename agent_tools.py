"""
Agent 工具集 — 搜索、计算、文件读写
"""
import math
import re
from pathlib import Path


# ═══════════════════════════════════════════════════════
#  计算器
# ═══════════════════════════════════════════════════════
def safe_calc(expression: str) -> str:
    """安全地计算数学表达式，支持 + - * / ** sqrt sin cos abs round"""
    allowed = {
        "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
        "abs": abs, "round": round, "pow": pow,
        "pi": math.pi, "e": math.e, "log": math.log, "log10": math.log10,
        "ceil": math.ceil, "floor": math.floor,
    }
    # 只允许安全字符
    sanitized = expression.strip()
    if not re.match(r'^[\d\s+\-*/().,%\^a-z_]+$', sanitized, re.IGNORECASE):
        return "错误：表达式包含不允许的字符"
    try:
        result = eval(sanitized, {"__builtins__": {}}, allowed)
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算错误：{e}"


# ═══════════════════════════════════════════════════════
#  网页搜索
# ═══════════════════════════════════════════════════════
def web_search(query: str, max_results: int = 3) -> str:
    """DuckDuckGo 即时搜索，返回摘要"""
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(f"**{r['title']}**\n{r['body']}\n🔗 {r.get('href', '')}")
        if not results:
            return "未找到相关结果。"
        return "\n\n---\n\n".join(results)
    except ImportError:
        return "⚠️ web_search 不可用（需安装 duckduckgo-search）"
    except Exception as e:
        return f"搜索出错：{e}"


# ═══════════════════════════════════════════════════════
#  文件操作
# ═══════════════════════════════════════════════════════
SANDBOX_DIR = Path(__file__).parent / "agent_workspace"
SANDBOX_DIR.mkdir(exist_ok=True)


def read_agent_file(filepath: str) -> str:
    """读取工作区文件内容"""
    path = (SANDBOX_DIR / filepath).resolve()
    if not str(path).startswith(str(SANDBOX_DIR.resolve())):
        return "错误：不允许访问外部路径"
    if not path.exists():
        return f"文件不存在：{filepath}"
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        if len(content) > 5000:
            content = content[:5000] + f"\n\n...（截断，共 {len(content)} 字符）"
        return content
    except Exception as e:
        return f"读取失败：{e}"


def write_agent_file(filepath: str, content: str) -> str:
    """写入文件到工作区"""
    path = (SANDBOX_DIR / filepath).resolve()
    if not str(path).startswith(str(SANDBOX_DIR.resolve())):
        return "错误：不允许访问外部路径"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"已写入：{filepath}（{len(content)} 字符）"
    except Exception as e:
        return f"写入失败：{e}"


def list_agent_files() -> str:
    """列出工作区所有文件"""
    try:
        files = []
        for f in sorted(SANDBOX_DIR.rglob("*")):
            if f.is_file():
                rel = f.relative_to(SANDBOX_DIR)
                size = f.stat().st_size
                files.append(f"  {rel} ({_fmt_size(size)})")
        return "\n".join(files) if files else "（空）"
    except Exception as e:
        return f"列出失败：{e}"


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB"):
        if n < 1024:
            return f"{n}{unit}"
        n //= 1024
    return f"{n}GB"


# ═══════════════════════════════════════════════════════
#  代码执行沙箱
# ═══════════════════════════════════════════════════════
import subprocess
import tempfile
import sys
import os as _os


def run_python(code: str) -> str:
    """在隔离子进程中执行 Python 代码，30 秒超时，返回 stdout/stderr"""
    # 写入临时文件
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    )
    tmp.write(code)
    tmp.close()

    try:
        proc = subprocess.run(
            [sys.executable, "-u", tmp.name],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(SANDBOX_DIR),
            env={**_os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        out = proc.stdout
        err = proc.stderr

        result_parts = []
        if out.strip():
            result_parts.append(f"[stdout]\n{out.rstrip()}")
        if err.strip():
            result_parts.append(f"[stderr]\n{err.rstrip()}")
        if not out.strip() and not err.strip():
            result_parts.append("(无输出)")

        output = "\n\n".join(result_parts)

        # 截断
        max_chars = 5000
        if len(output) > max_chars:
            output = output[:max_chars] + f"\n\n...（截断，共 {len(output)} 字符）"

        return f"✅ 退出码 {proc.returncode}\n\n{output}"

    except subprocess.TimeoutExpired:
        return "⏱️ 执行超时（30 秒），代码可能包含死循环或耗时操作"

    except Exception as e:
        return f"❌ 执行失败：{e}"

    finally:
        try:
            Path(tmp.name).unlink(missing_ok=True)
        except Exception:
            pass

# ═══════════════════════════════════════════════════════
#  LangChain 工具包装
# ═══════════════════════════════════════════════════════
def get_langchain_tools():
    """返回 LangChain Tool 列表"""
    from langchain_core.tools import tool

    @tool
    def calculator(expression: str) -> str:
        """计算数学表达式。输入如 '2+3*4' 或 'sqrt(144)'，支持 + - * / ** sqrt sin cos abs round log pi e"""
        return safe_calc(expression)

    @tool
    def search_web(query: str) -> str:
        """搜索互联网获取最新信息。输入要搜索的关键词或问题"""
        return web_search(query)

    @tool
    def list_files(_: str = "") -> str:
        """列出 Agent 工作区中的所有文件"""
        return list_agent_files()

    @tool
    def read_file(filepath: str) -> str:
        """读取 Agent 工作区中的文件。输入相对于工作区的文件路径"""
        return read_agent_file(filepath)

    @tool
    def write_file(input_str: str) -> str:
        """写入文件到 Agent 工作区。输入格式：'文件路径|文件内容'"""
        parts = input_str.split("|", 1)
        if len(parts) != 2:
            return "格式错误。正确格式：'文件路径|文件内容'"
        return write_agent_file(parts[0].strip(), parts[1].strip())

    @tool
    def fetch_webpage(url: str) -> str:
        """抓取网页内容。输入完整 URL（如 https://example.com），返回页面标题和正文"""
        from browser import fetch_page_content
        # 需要 API key 做摘要 — 从环境读取
        import os
        from pathlib import Path as _Path
        from dotenv import load_dotenv as _load

        _load()
        key = os.getenv("ZHIPU_API_KEY", "")
        base = "https://open.bigmodel.cn/api/paas/v4/"
        result = fetch_page_content(url, key, base)
        if "error" in result:
            return f"❌ {result['error']}"
        parts = [f"📄 {result['title']}", f"🔗 {result['url']}", ""]
        if result.get("summary"):
            parts.append(f"**AI 摘要：**\n{result['summary']}")
            parts.append("\n---\n**正文：**")
        parts.append(result["text"][:3000])
        return "\n".join(parts)

    @tool
    def python_exec(code: str) -> str:
        """执行 Python 代码并返回结果。代码会保存到临时文件、用 subprocess 隔离运行、30 秒超时。输入完整的 Python 代码"""
        return run_python(code)

    return [calculator, search_web, fetch_webpage, python_exec, list_files, read_file, write_file]
