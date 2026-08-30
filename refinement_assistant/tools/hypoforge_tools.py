"""HypoForge refinement-specific tools used by the local agent workbench."""

from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

# 最近一次 save_refinement_output 保存的 run_id，供会话自动关联细化产出
LAST_SAVED_REFINEMENT_RUN_ID = ""


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default) or default


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _serper_search(query: str, api_key: str, limit: int = 6) -> str:
    payload = json.dumps({"q": query, "num": limit}).encode("utf-8")
    req = urllib.request.Request(
        "https://google.serper.dev/search",
        data=payload,
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    results = (data.get("organic") or [])[:limit]
    lines = []
    for item in results:
        title = str(item.get("title") or "").strip()
        link = str(item.get("link") or "").strip()
        snippet = str(item.get("snippet") or "").strip()
        lines.append(f"- {title}\n  {link}\n  {snippet}")
    return "\n".join(lines) if lines else "No web results."


def _ddg_search(query: str, limit: int = 6) -> str:
    url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    blocks = re.findall(
        r'<a rel="nofollow" class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        html,
        flags=re.S | re.I,
    )
    lines = []
    for href, title in blocks[:limit]:
        title = re.sub(r"<[^>]+>", "", title).strip()
        if "uddg=" in href:
            href = urllib.parse.unquote(href.split("uddg=", 1)[1].split("&", 1)[0])
        lines.append(f"- {title}\n  {href}")
    if not lines:
        return "No web results (DuckDuckGo HTML parse failed)."
    return "\n".join(lines)


def web_search(query: str, limit: int = 6) -> str:
    key = _env("SERPER_API_KEY")
    try:
        if key:
            return _serper_search(query, key, limit)
        return _ddg_search(query, limit)
    except Exception as exc:
        return f"Web search failed: {type(exc).__name__}: {exc}"


def read_downloaded_papers(run_id: str = "", limit: int = 20) -> str:
    root = _project_root()
    candidates = []
    if run_id:
        candidates.append(root / "output" / "ui_runs_7862" / run_id)
        candidates.append(root / "output" / "ui_runs" / run_id)
        candidates.append(root / "output" / "refined" / run_id)
    candidates.append(root / "knowledge_graph" / "papers.jsonl")

    seen = set()
    lines = []
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".jsonl":
            try:
                with path.open("r", encoding="utf-8") as f:
                    for idx, raw in enumerate(f):
                        if idx >= limit:
                            break
                        raw = raw.strip()
                        if not raw:
                            continue
                        try:
                            item = json.loads(raw)
                        except Exception:
                            continue
                        title = str(
                            item.get("title") or item.get("name") or item.get("paper_id") or ""
                        )
                        abstract = str(item.get("abstract") or item.get("summary") or "")[:500]
                        if title and title not in seen:
                            seen.add(title)
                            lines.append(f"## {title}\n{abstract}")
            except Exception:
                continue
        elif path.is_dir():
            for f in sorted(path.rglob("*.json")):
                if len(lines) >= limit:
                    break
                try:
                    data = json.loads(f.read_text(encoding="utf-8", errors="ignore"))
                except Exception:
                    continue
                texts = []
                for key in ("input_question", "original_question"):
                    val = data.get(key)
                    if val:
                        texts.append(str(val)[:300])
                for hyp in (data.get("top_hypotheses") or [])[:3]:
                    texts.append("Hypothesis: " + str(hyp.get("statement") or "")[:300])
                if texts:
                    lines.append("## " + f.name + "\n" + "\n".join(texts))
    if not lines:
        return "No downloaded papers found. You may need to provide a HypoForge run_id first."
    return "\n\n".join(lines)


def save_refinement_output(run_id: str, json_data: str, markdown: str = "") -> str:
    global LAST_SAVED_REFINEMENT_RUN_ID
    root = _project_root()
    run_id = (run_id or "").strip() or datetime.now().strftime("refine-%Y%m%d-%H%M%S")
    out_dir = root / "output" / "refined" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        parsed = json.loads(json_data) if json_data.strip() else {}
    except Exception:
        parsed = {"raw": json_data}
    json_path = out_dir / "refinement.json"
    md_path = out_dir / "refinement.md"
    json_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(markdown or str(parsed), encoding="utf-8")
    # 记录最近一次保存，供会话自动关联（手动开头的对话没有 HypoForge run_id）
    LAST_SAVED_REFINEMENT_RUN_ID = run_id
    return f"Refinement saved:\nJSON: {json_path}\nMarkdown: {md_path}"


def generate_dry_lab_script(domain: str, task: str = "", detail: str = "") -> str:
    domain = (domain or "").strip().casefold()
    task = (task or "").strip()
    if "protein" in domain or "foldx" in domain or "bio" in domain:
        script = f'''# Dry-lab starter: protein / FoldX oriented
# Target: {task or "protein stability / interaction analysis"}
import subprocess
from pathlib import Path

PDB = Path("input.pdb")
FOLDX = Path("FoldX")
if not PDB.exists():
    raise SystemExit(f"Missing PDB: {{PDB}}")
if not FOLDX.exists():
    print("FoldX not found; install and license it first.")
cmd = [str(FOLDX), "--command=RepairPDB", "--pdb=" + PDB.stem]
print("Planned command:", " ".join(cmd))
# subprocess.run(cmd, check=False)
print("Dry-lab script generated. Run only after permission.")
'''
    elif "ai" in domain or "computer" in domain or "vision" in domain or "point" in domain or "3d" in domain:
        script = f'''# Dry-lab starter: AI / computer-vision oriented
# Task: {task or "multiview image to 3D point cloud (or similar)"}
import numpy as np
def make_synthetic_point_cloud(n: int = 1000) -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.stack([rng.normal(0,1,n), rng.normal(0,1,n), rng.normal(0,1,n)], axis=1)
points = make_synthetic_point_cloud()
print("Generated synthetic point cloud:", points.shape)
print("Dry-lab script generated. Run only after permission.")
'''
    else:
        script = f'''# Dry-lab starter: generic computational verification
# Task: {task or "computational analysis or simulation"}
import numpy as np
data = np.random.default_rng(0).normal(size=(100, 3))
print("Synthetic data shape:", data.shape)
print("Dry-lab script generated. Run only after permission.")
'''
    return script
