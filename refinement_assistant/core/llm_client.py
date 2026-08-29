import os
import json
import time
import hashlib
import sqlite3
import urllib.error
import urllib.request
from contextlib import closing
from types import SimpleNamespace
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
# 默认读取仓库根目录的 .env（与 HypoForge 主程序共用），可用 HYPOFORGE_ENV 覆盖
_repo_root = Path(__file__).resolve().parents[2]
_hypo_env = Path(os.getenv("HYPOFORGE_ENV", str(_repo_root / ".env")))
if _hypo_env.exists():
    load_dotenv(_hypo_env, override=True)


def _to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(v) for v in value]
    return value


def _to_plain(value):
    if isinstance(value, SimpleNamespace):
        return {k: _to_plain(v) for k, v in value.__dict__.items() if v is not None}
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_to_plain(v) for v in value]
    return value


class DeepSeekCompletions:
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def create(self, **kwargs):
        if not self.api_key:
            raise RuntimeError("QWEN_API_KEY is not configured")

        payload = {key: _to_plain(value) for key, value in kwargs.items() if value is not None}
        cache_key = _response_cache_key(payload)
        if cache_key:
            cached = _read_response_cache(cache_key)
            if cached is not None:
                return _to_namespace(_as_local_cache_hit(cached))

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_error = None
        # 网络抖动或 API 端挂起时自动重试（4xx 除 429 外不重试）
        for attempt in range(3):
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                # 长计划生成可能超过 90s，放宽到 300s
                with urllib.request.urlopen(request, timeout=300) as response:
                    raw = response.read().decode("utf-8")
                if cache_key:
                    _write_response_cache(cache_key, raw)
                return _to_namespace(json.loads(raw))
            except urllib.error.HTTPError as e:
                error_body = e.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"DeepSeek API error {e.code}: {error_body}")
                if e.code < 500 and e.code != 429:
                    raise last_error
            except (urllib.error.URLError, OSError) as e:
                reason = getattr(e, "reason", e)
                last_error = RuntimeError(f"API connection error: {reason}")
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
                print(f">> LLM 调用失败（{last_error}），第 {attempt + 2} 次重试...")
        raise last_error


class DeepSeekChat:
    def __init__(self, api_key, base_url):
        self.completions = DeepSeekCompletions(api_key, base_url)


class DeepSeekClient:
    def __init__(self, api_key, base_url):
        self.chat = DeepSeekChat(api_key, base_url)


def create_llm_client():
    return DeepSeekClient(
        api_key=os.getenv("QWEN_API_KEY"),
        base_url=os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )


def get_chat_model():
    return os.getenv("QWEN_MODEL", "qwen3.7-plus")


def get_vision_model():
    return os.getenv("QWEN_VISION_MODEL", "")


def is_image_input_enabled():
    return os.getenv("QWEN_ENABLE_IMAGE_INPUT", "False").lower() in {"1", "true", "yes", "on"}


def message_to_dict(message):
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    if hasattr(message, "to_dict"):
        return message.to_dict()
    if isinstance(message, SimpleNamespace):
        return _to_plain(message)
    return dict(message)


def _response_cache_enabled():
    return os.getenv("QWEN_RESPONSE_CACHE", "1").lower() in {"1", "true", "yes", "on"}


def _response_cache_path():
    configured = os.getenv("QWEN_CACHE_PATH")
    if configured:
        return Path(configured)
    workspace = os.getenv("WORKSPACE")
    if workspace:
        return Path(workspace) / "llm_response_cache.sqlite3"
    return Path.cwd() / "data" / "llm_response_cache.sqlite3"


def _response_cache_key(payload):
    if not _response_cache_enabled():
        return None
    if payload.get("stream") or "tools" in payload or "tool_choice" in payload:
        return None
    if not payload.get("messages"):
        return None
    stable = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()


def _connect_response_cache():
    path = _response_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_response_cache (
            cache_key TEXT PRIMARY KEY,
            response_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            hit_count INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    return conn


def _read_response_cache(cache_key):
    try:
        with closing(_connect_response_cache()) as conn:
            with conn:
                row = conn.execute(
                    "SELECT response_json FROM llm_response_cache WHERE cache_key = ?",
                    (cache_key,),
                ).fetchone()
                if not row:
                    return None
                conn.execute(
                    "UPDATE llm_response_cache SET hit_count = hit_count + 1 WHERE cache_key = ?",
                    (cache_key,),
                )
                return json.loads(row[0])
    except Exception:
        return None


def _write_response_cache(cache_key, response_json):
    try:
        json.loads(response_json)
        with closing(_connect_response_cache()) as conn:
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO llm_response_cache (cache_key, response_json)
                    VALUES (?, ?)
                    """,
                    (cache_key, response_json),
                )
    except Exception:
        return


def _as_local_cache_hit(response_data):
    cloned = json.loads(json.dumps(response_data, ensure_ascii=False))
    original_usage = cloned.get("usage") or {}
    cloned["usage"] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "local_cache_hit": True,
        "cached_prompt_tokens": original_usage.get("prompt_tokens", 0),
        "cached_completion_tokens": original_usage.get("completion_tokens", 0),
        "cached_total_tokens": original_usage.get("total_tokens", 0),
    }
    return cloned
