# token_manager.py
import json
import os
from logger_manager import logger

# 默认额度：100 万 token（可通过环境变量 MAX_TOTAL_TOKEN 或设置界面调整）
DEFAULT_TOKEN_LIMIT = 1_000_000
# 模型上下文窗口（qwen-plus 级别 128k），用于判断"上下文压力"，与累计额度无关
DEFAULT_CONTEXT_WINDOW = 131_072


class TokenManager:
    """同时管理两个维度：
    - max_threshold: 会话累计消耗预算（额度），超过后停止迭代
    - context_window: 模型单次请求的上下文窗口，last_prompt_tokens 接近上限时需要压缩
    累计用量持久化到 token_state.json，服务重启后不清零。
    """

    def __init__(self, default_limit=None):
        limit = default_limit if default_limit else os.getenv("MAX_TOTAL_TOKEN", DEFAULT_TOKEN_LIMIT)
        self.max_threshold = int(limit)
        self.context_window = int(os.getenv("QWEN_CONTEXT_WINDOW", DEFAULT_CONTEXT_WINDOW))
        self.total_used = 0
        self.last_prompt_tokens = 0
        self._state_path = os.path.join(
            os.getenv("WORKSPACE") or os.path.dirname(os.path.abspath(__file__)),
            "token_state.json",
        )
        self._load_state()

    def _load_state(self):
        try:
            with open(self._state_path, encoding="utf-8") as f:
                state = json.load(f)
            self.total_used = int(state.get("total_used", 0))
            self.last_prompt_tokens = int(state.get("last_prompt_tokens", 0))
        except Exception:
            pass

    def _save_state(self):
        try:
            with open(self._state_path, "w", encoding="utf-8") as f:
                json.dump({
                    "total_used": self.total_used,
                    "last_prompt_tokens": self.last_prompt_tokens,
                    "max_threshold": self.max_threshold,
                }, f)
        except Exception:
            pass

    def update_usage(self, response, source="General"):
        """记录 Token 用量，包括 API 侧缓存的命中/未命中"""
        usage = response.usage
        p_tokens = usage.prompt_tokens
        c_tokens = usage.completion_tokens
        t_tokens = usage.total_tokens

        cache_hit = getattr(usage, 'prompt_cache_hit_tokens', 0)
        cache_miss = getattr(usage, 'prompt_cache_miss_tokens', 0)
        local_cache_hit = getattr(usage, 'local_cache_hit', False)

        self.total_used += t_tokens
        if p_tokens:
            # 记录最近一次请求的真实上下文长度，作为压缩决策依据
            self.last_prompt_tokens = max(self.last_prompt_tokens, p_tokens)
        self._save_state()

        log_msg = (f"[TOKEN审计] 来源: {source:<20} | 本次: {t_tokens} (P:{p_tokens}, C:{c_tokens}) | "
                   f"总用量: {self.total_used}/{self.max_threshold}")
        if local_cache_hit:
            saved = getattr(usage, 'cached_total_tokens', 0)
            log_msg += f" | local cache hit: saved about {saved} tokens"
        if cache_hit or cache_miss:
            hit_rate = cache_hit / (cache_hit + cache_miss) * 100 if (cache_hit + cache_miss) > 0 else 0
            log_msg += f" | 💾缓存 命中:{cache_hit} 未命中:{cache_miss} ({hit_rate:.0f}%)"
        logger.info(log_msg)

        if self.total_used >= self.max_threshold:
            err_msg = f"Token 额度已耗尽 ({self.total_used}/{self.max_threshold})"
            logger.critical(err_msg)
            return False, err_msg

        return True, "Success"

    def context_pressure(self):
        """最近一次请求的上下文占用比例 (0.0 - 1.0+)"""
        if self.context_window <= 0:
            return 0.0
        return self.last_prompt_tokens / self.context_window

    def is_over_limit(self):
        return self.total_used >= self.max_threshold

    def get_status_report(self):
        return f"[{self.total_used} / {self.max_threshold}]"

    def reset_usage(self):
        self.total_used = 0
        self.last_prompt_tokens = 0
        self._save_state()

    def restore_usage(self, used):
        """切换/加载会话时恢复该会话自己的累计用量"""
        try:
            self.total_used = max(0, int(used))
        except (TypeError, ValueError):
            self.total_used = 0
        self.last_prompt_tokens = 0
        self._save_state()

    def double_threshold(self):
        self.max_threshold *= 2

    def set_threshold(self, new_limit):
        limit = max(1000, min(int(new_limit), 10_000_000))
        self.max_threshold = limit
        self._save_state()
        logger.info(f"[TOKEN] 阈值已调整为: {self.max_threshold}")
