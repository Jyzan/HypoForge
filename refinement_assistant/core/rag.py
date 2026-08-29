# core/rag.py
"""轻量级工作区知识检索（RAG）。

对 HypoForge 的论文、固定方案、细化产出和会话记忆建立本地分块索引，
供细化 Agent 通过 search_workspace_knowledge 工具按需检索相关片段，
替代"整篇读文件"的粗放方式。

检索策略：
- 默认：零依赖关键词检索（CJK 二元组 + ASCII 词，IDF 加权），离线可用。
- 可选：设置 QWEN_EMBEDDING_MODEL（如 text-embedding-v4）后自动启用向量检索，
  与关键词得分混合；向量调用失败或未配置时回退纯关键词。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.request
from pathlib import Path

from logger_manager import logger

# 索引规模保护
MAX_FILES_PER_SOURCE = 50
MAX_TEXT_PER_FILE = 20_000
MAX_CHUNKS = 800
MAX_EMBED_CHUNKS = 600
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
EMBED_BATCH = 10

_ASCII_WORD_RE = re.compile(r"[a-zA-Z0-9_]{2,}")


def _tokenize(text):
    """CJK 拆二元组，ASCII 取小写词，统一为词袋"""
    tokens = []
    for word in _ASCII_WORD_RE.findall(text):
        tokens.append(word.lower())
    for match in re.finditer(r"[\u4e00-\u9fff]+", text):
        run = match.group()
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


class WorkspaceKnowledgeIndex:
    def __init__(self, assistant_dir, project_root):
        self.assistant_dir = Path(assistant_dir)
        self.project_root = Path(project_root)
        self.chunks = []          # [{"source", "title", "text", "tokens", "vector"}]
        self._df = {}             # 词文档频率
        self._fingerprint = ""
        self._embed_model = os.getenv("QWEN_EMBEDDING_MODEL", "")
        self._embed_enabled = bool(self._embed_model and os.getenv("QWEN_API_KEY"))

    # ---------- 索引构建 ----------

    def _collect_files(self):
        """返回 [(title, path, extractor)]，extractor 决定该文件如何抽取文本"""
        root, assistant = self.project_root, self.assistant_dir
        files = []

        def add(path, extractor="raw"):
            if path.exists() and path not in [p for _, p, _ in files]:
                files.append((path.name, path, extractor))

        # 固定方案（细化任务输入）
        for cand in [assistant / "refinement_input.json",
                     root / "output" / "refined" / "refinement_input.json"]:
            add(cand, "raw")

        # 论文库（HypoForge 文献检索产物）
        add(root / "knowledge_graph" / "papers.jsonl", "papers_jsonl")

        # 各运行目录的主结果文件（含原始问题与假设）
        run_files = []
        for run_dir_pattern in ("ui_runs", "ui_runs_7862"):
            base = root / "output" / run_dir_pattern
            if not base.exists():
                continue
            for run_dir in base.iterdir():
                if not run_dir.is_dir():
                    continue
                main_file = run_dir / (run_dir.name + ".json")
                if main_file.exists():
                    run_files.append(main_file)
        run_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for path in run_files[:MAX_FILES_PER_SOURCE]:
            add(path, "run_json")

        # 细化产出
        refined = root / "output" / "refined"
        if refined.exists():
            out_files = sorted(
                (p for p in refined.rglob("*") if p.suffix in (".md", ".json") and p.is_file()),
                key=lambda p: p.stat().st_mtime, reverse=True)
            for path in out_files[:MAX_FILES_PER_SOURCE]:
                add(path, "raw")

        # 会话记忆与技能文档
        for sub in ("memories", "skills"):
            base = assistant / sub
            if base.exists():
                for path in sorted(base.glob("*.md")):
                    add(path, "raw")

        return files

    @staticmethod
    def _extract_text(path, extractor):
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")[:MAX_TEXT_PER_FILE]
            if extractor == "raw":
                return raw
            if extractor == "papers_jsonl":
                parts = []
                for line in raw.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except Exception:
                        continue
                    paper = item.get("paper") or item
                    title = paper.get("title") or paper.get("name") or ""
                    abstract = paper.get("abstract") or paper.get("summary") or ""
                    if title:
                        parts.append(f"{title}\n{abstract}"[:3000])
                return "\n\n".join(parts)
            if extractor == "run_json":
                try:
                    data = json.loads(raw or "{}")
                except Exception:
                    return ""
                parts = []
                for key in ("input_question", "original_question"):
                    if data.get(key):
                        parts.append(str(data[key])[:2000])
                for hyp in (data.get("top_hypotheses") or [])[:5]:
                    stmt = hyp.get("statement") if isinstance(hyp, dict) else hyp
                    if stmt:
                        parts.append("Hypothesis: " + str(stmt)[:800])
                return "\n\n".join(parts)
        except Exception as exc:
            logger.warning(f"[RAG] 抽取失败 {path.name}: {exc}")
        return ""

    @staticmethod
    def _chunk_text(text):
        """按段落聚合为固定大小块，带重叠"""
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        chunks, buf = [], ""
        for para in paragraphs:
            if len(buf) + len(para) + 1 > CHUNK_SIZE and buf:
                chunks.append(buf)
                buf = buf[-CHUNK_OVERLAP:] if CHUNK_OVERLAP else ""
            buf += ("\n" + para if buf else para)
        if buf.strip():
            chunks.append(buf)
        # 超长单块硬切
        final = []
        for chunk in chunks:
            for i in range(0, len(chunk), CHUNK_SIZE):
                piece = chunk[i:i + CHUNK_SIZE]
                if piece.strip():
                    final.append(piece)
        return final

    def _fingerprint_of(self, files):
        hasher = hashlib.sha1()
        for _, path, _ in sorted(files, key=lambda f: str(f[1])):
            try:
                stat = path.stat()
                hasher.update(f"{path}|{stat.st_mtime_ns}|{stat.st_size}".encode())
            except OSError:
                pass
        return hasher.hexdigest()

    def build(self, force=False):
        files = self._collect_files()
        fingerprint = self._fingerprint_of(files)
        if not force and fingerprint == self._fingerprint and self.chunks:
            return
        self._fingerprint = fingerprint

        chunks = []
        for title, path, extractor in files:
            text = self._extract_text(path, extractor)
            for piece in self._chunk_text(text):
                chunks.append({
                    "source": str(path),
                    "title": title,
                    "text": piece,
                    "tokens": _tokenize(f"{title} {piece}"),
                    "vector": None,
                })
                if len(chunks) >= MAX_CHUNKS:
                    break
            if len(chunks) >= MAX_CHUNKS:
                break

        self.chunks = chunks
        self._build_df()
        if self._embed_enabled:
            self._embed_pending()
        logger.info(f"[RAG] 索引就绪: {len(self.chunks)} 块 / {len(files)} 文件 "
                    f"(embedding={'on' if self._embed_enabled else 'off'})")

    def _build_df(self):
        self._df = {}
        for chunk in self.chunks:
            for token in set(chunk["tokens"]):
                self._df[token] = self._df.get(token, 0) + 1

    # ---------- 向量检索（可选） ----------

    def _embed_texts(self, texts):
        base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
        api_key = os.getenv("QWEN_API_KEY")
        payload = json.dumps({
            "model": self._embed_model,
            "input": texts,
            "dimensions": 1024,
            "encoding_format": "float",
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/embeddings",
            data=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        items = sorted(data.get("data", []), key=lambda d: d.get("index", 0))
        return [item["embedding"] for item in items]

    def _embed_pending(self):
        pending = [c for c in self.chunks if c["vector"] is None][:MAX_EMBED_CHUNKS]
        done = 0
        for i in range(0, len(pending), EMBED_BATCH):
            batch = pending[i:i + EMBED_BATCH]
            try:
                vectors = self._embed_texts([c["text"] for c in batch])
            except Exception as exc:
                logger.warning(f"[RAG] 向量批量失败，本批回退关键词: {exc}")
                return
            for chunk, vec in zip(batch, vectors):
                chunk["vector"] = vec
            done += len(batch)
        if done:
            logger.info(f"[RAG] 已向量化 {done} 块")

    # ---------- 检索 ----------

    def _keyword_scores(self, query_tokens):
        n = max(1, len(self.chunks))
        scores = []
        for chunk in self.chunks:
            tf = {}
            for token in chunk["tokens"]:
                tf[token] = tf.get(token, 0) + 1
            score = 0.0
            for token in query_tokens:
                if token not in tf:
                    continue
                idf = math.log(1 + n / (1 + self._df.get(token, 0)))
                score += idf * (1 + math.log(tf[token]))
            scores.append(score / (1 + len(chunk["tokens"]) ** 0.5))  # 长度归一
        return scores

    @staticmethod
    def _cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (na * nb)

    def search(self, query, top_k=5, rebuild=False):
        if rebuild:
            self._fingerprint = ""
            self.build(force=True)
        elif not self.chunks:
            self.build()

        if not self.chunks:
            return "知识库为空：尚未找到可索引的方案、论文或产出文件。"

        query_tokens = _tokenize(query)
        kw_scores = self._keyword_scores(query_tokens)

        vectors = None
        if self._embed_enabled and any(c["vector"] is not None for c in self.chunks):
            try:
                vectors = self._embed_texts([query])[0]
            except Exception as exc:
                logger.warning(f"[RAG] 查询向量化失败，回退关键词: {exc}")

        ranked = []
        for i, chunk in enumerate(self.chunks):
            score = kw_scores[i]
            if vectors is not None and chunk["vector"] is not None:
                # 混合：向量相似为主，关键词为辅
                score = 0.7 * (self._cosine(vectors, chunk["vector"]) + 1) / 2 + 0.3 * min(score, 1.0)
            ranked.append((score, i))
        ranked.sort(reverse=True)

        top = [(s, i) for s, i in ranked[:max(1, int(top_k))] if s > 0]
        if not top:
            return "知识库中没有检索到相关内容。可尝试换个关键词，或用 web_search 搜索互联网。"

        lines = [f"检索到 {len(top)} 个相关片段（来源: 论文/方案/细化产出本地索引）:"]
        for rank, (score, idx) in enumerate(top, 1):
            chunk = self.chunks[idx]
            lines.append(f"\n[{rank}] score={score:.3f} | {chunk['title']}\n来源: {chunk['source']}\n{chunk['text']}")
        return "\n".join(lines)


_SINGLETON = None


def get_workspace_index():
    """进程级单例：assistant 目录可用 WORKSPACE 覆盖，项目根为仓库根"""
    global _SINGLETON
    if _SINGLETON is None:
        assistant_dir = os.getenv("WORKSPACE") or str(Path(__file__).resolve().parents[1])
        project_root = Path(__file__).resolve().parents[2]
        _SINGLETON = WorkspaceKnowledgeIndex(assistant_dir, project_root)
    return _SINGLETON
