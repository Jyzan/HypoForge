# config.py
import os
import json
from logger_manager import logger

class ConfigManager:
    def __init__(self, workspace_path):
        self.config_file = os.path.join(workspace_path, "config.json")
        # 默认配置模板
        self.config = {
            "enable_allowlist": True,
            "disabled_tools": [],       # 例如 ["mouse_click", "get_weather"]
            "disabled_skills": [],      # 例如 ["powershell_tips.md"]
            "sandbox_confirmed": False,
            "sandbox_path": "",
            "token_limit": 1000000,     # 会话累计 Token 预算，可在设置界面调整
            "full_trust_mode": False    # 完全信任模式：跳过所有权限审批（慎用）
        }
        self.load()

    def load(self):
        """加载配置，如果不存在则自动生成默认配置"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    self.config.update(loaded_config)
            except Exception as e:
                logger.error(f"[CONFIG] 读取配置文件失败: {e}")
        else:
            self.save()

    def save(self):
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=4)
            logger.info("[CONFIG] 配置文件已生成/保存。")
        except Exception as e:
            logger.error(f"[CONFIG] 保存配置文件失败: {e}")

    def is_tool_enabled(self, tool_name):
        return tool_name not in self.config.get("disabled_tools", [])

    def is_skill_enabled(self, skill_name):
        return skill_name not in self.config.get("disabled_skills", [])

    def is_allowlist_enabled(self):
        return self.config.get("enable_allowlist", True)
