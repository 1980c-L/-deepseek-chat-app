"""
内置浏览器模块 — 抓取网页内容供 AI 分析
"""
import re
from urllib.parse import urlparse
import openai


def fetch_page_content(url: str, api_key: str, base_url: str) -> dict:
    """
    抓取网页 → 提取正文 → AI 摘要
    返回 {"url": ..., "title": ..., "text": ..., "summary": ...}
    """
    import urllib.request
    import urllib.error

    # 1. 抓取 HTML
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
            },
        )
        resp = urllib.request.urlopen(req, timeout=15)
        html = resp.read().decode(resp.headers.get_content_charset() or "utf-8", errors="replace")
        final_url = resp.geturl()
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.reason}"}
    except Exception as e:
        return {"error": f"无法访问：{e}"}

    # 2. 提取标题
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    title = title_match.group(1).strip() if title_match else urlparse(final_url).netloc

    # 3. 提取正文（去掉 script/style/head/nav/footer）
    for tag in ["script", "style", "head", "nav", "footer", "header", "aside", "noscript", "iframe"]:
        html = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", "", html, flags=re.IGNORECASE | re.DOTALL)

    # 去掉 HTML 标签
    text = re.sub(r"<[^>]+>", " ", html)
    # 合并空白 + 去重换行
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    # 截断到 8000 字符
    if len(text) > 8000:
        text = text[:8000] + f"\n\n...（截断，共 {len(text)} 字符）"

    if not text.strip() or len(text) < 50:
        return {"error": "无法提取有效正文", "title": title, "url": final_url}

    # 4. AI 摘要
    summary = ""
    if api_key and len(text) > 200:
        try:
            client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=20)
            resp = client.chat.completions.create(
                model="GLM-4-Flash-250414",
                messages=[{
                    "role": "user",
                    "content": f"请用 3-5 句话中文摘要以下网页内容，突出重点：\n\n{text[:4000]}",
                }],
                max_tokens=300,
                temperature=0.3,
            )
            summary = resp.choices[0].message.content.strip()
        except Exception:
            summary = "（摘要生成失败）"

    return {
        "url": final_url,
        "title": title,
        "text": text,
        "summary": summary,
    }


def extract_readable_text(html: str) -> str:
    """快速提取 HTML 中的可读文本（给 Agent tool 用）"""
    for tag in ["script", "style", "nav", "footer", "header", "aside", "noscript"]:
        html = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", "", html, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()[:5000]
