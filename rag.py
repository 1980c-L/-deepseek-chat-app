"""
RAG 文档问答模块 — 解析、切片、向量化、检索
"""
import hashlib
import json
import re
from pathlib import Path
from typing import Optional

# 懒加载，不阻塞启动
_pdf_reader = None
_docx_reader = None
_text_splitter = None
_faiss = None
_numpy = None


def _ensure_pdf():
    global _pdf_reader
    if _pdf_reader is None:
        from pypdf import PdfReader
        _pdf_reader = PdfReader


def _ensure_docx():
    global _docx_reader
    if _docx_reader is None:
        from docx import Document
        _docx_reader = Document


def _ensure_splitter():
    global _text_splitter
    if _text_splitter is None:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        _text_splitter = RecursiveCharacterTextSplitter


def _ensure_faiss():
    global _faiss, _numpy
    if _faiss is None:
        import faiss
        import numpy as np
        _faiss = faiss
        _numpy = np


def parse_document(file_bytes: bytes, filename: str) -> str:
    """解析 PDF / DOCX / TXT → 纯文本"""
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        _ensure_pdf()
        import io
        reader = _pdf_reader(io.BytesIO(file_bytes))
        return "\n\n".join(
            page.extract_text() or "" for page in reader.pages
        )
    elif ext in (".docx", ".doc"):
        _ensure_docx()
        import io
        doc = _docx_reader(io.BytesIO(file_bytes))
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    elif ext in (".txt", ".md", ".py", ".json", ".yaml", ".yml", ".csv"):
        return file_bytes.decode("utf-8", errors="replace")
    else:
        raise ValueError(f"不支持的文件格式：{ext}")


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 80) -> list[str]:
    """文本切片"""
    _ensure_splitter()
    splitter = _text_splitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
    )
    chunks = splitter.split_text(text)
    # 过滤纯空白
    return [c.strip() for c in chunks if c.strip() and len(c.strip()) > 20]


def embed_texts(texts: list[str], api_key: str, base_url: str) -> list[list[float]]:
    """用智谱 embedding-3 向量化"""
    import openai
    client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=30)
    response = client.embeddings.create(
        model="embedding-3",
        input=texts,
    )
    return [d.embedding for d in response.data]


def build_index(chunks: list[str], vectors: list[list[float]]) -> tuple:
    """构建 FAISS 索引，返回 (index, chunks)"""
    _ensure_faiss()
    dim = len(vectors[0])
    index = _faiss.IndexFlatIP(dim)  # 内积 = 余弦相似度（已归一化时）
    arr = _numpy.array(vectors, dtype="float32")
    # L2 归一化 → 内积 = 余弦相似度
    _faiss.normalize_L2(arr)
    index.add(arr)
    return index, chunks


def search(index, chunks: list[str], query_vec: list[float], top_k: int = 5) -> list[dict]:
    """检索 top_k 相关片段"""
    _ensure_faiss()
    q = _numpy.array([query_vec], dtype="float32")
    _faiss.normalize_L2(q)
    scores, indices = index.search(q, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0 and idx < len(chunks) and score > 0.3:  # 相似度阈值
            results.append({"content": chunks[idx], "score": float(score)})
    return results


class DocumentStore:
    """管理文档 → 向量索引的完整生命周期"""

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index = None
        self.chunks: list[str] = []
        self.doc_names: list[str] = []
        self._loaded = False

    @property
    def is_ready(self) -> bool:
        return self.index is not None and len(self.chunks) > 0

    def _doc_hash(self, filename: str, content: bytes) -> str:
        return hashlib.sha256(filename.encode() + content).hexdigest()[:16]

    def add_document(self, filename: str, file_bytes: bytes, api_key: str, base_url: str):
        """解析 + 切片 + 向量化 + 入库"""
        text = parse_document(file_bytes, filename)
        if not text.strip():
            return 0

        chunks = chunk_text(text)
        if not chunks:
            return 0

        vectors = embed_texts(chunks, api_key, base_url)
        if not vectors:
            return 0

        if self.index is None:
            self.index, self.chunks = build_index(chunks, vectors)
        else:
            # 追加到已有索引
            _ensure_faiss()
            dim = len(vectors[0])
            arr = _numpy.array(vectors, dtype="float32")
            _faiss.normalize_L2(arr)
            self.index.add(arr)
            self.chunks.extend(chunks)

        self.doc_names.append(filename)
        return len(chunks)

    def query(self, question: str, api_key: str, base_url: str, top_k: int = 5) -> str:
        """检索相关片段 → 拼成上下文"""
        if not self.is_ready:
            return ""

        qv = embed_texts([question], api_key, base_url)
        if not qv:
            return ""

        results = search(self.index, self.chunks, qv[0], top_k=top_k)
        if not results:
            return ""

        ctx_parts = []
        for i, r in enumerate(results, 1):
            ctx_parts.append(f"[片段 {i}] (相关度 {r['score']:.2f})\n{r['content']}")
        return "\n\n---\n\n".join(ctx_parts)

    def clear(self):
        self.index = None
        self.chunks = []
        self.doc_names = []
