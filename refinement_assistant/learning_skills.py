# learning_skills.py
import os
from logger_manager import logger

class SkillManager:
    def __init__(self, workspace_path, config):
        self.workspace_path = os.path.normpath(workspace_path)
        self.skill_dir = os.path.join(self.workspace_path, "skills")
        self.allowlist_dir = os.path.join(self.workspace_path, "allowlist")
        self.config = config # 注入配置管理器

    def _read_file_safe(self, file_path):
        encodings = ['utf-8', 'utf-8-sig', 'utf-16', 'gbk']
        for enc in encodings:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    return f.read()
            except:
                continue
        return None

    def load_core_memory(self):
        """加载核心记忆，并动态过滤掉被禁用的技能目录"""
        combined_content = []
        if not os.path.exists(self.skill_dir):
            os.makedirs(self.skill_dir)
            return "【技能库目前为空】"

        # 1. 加载 base.md (始终加载)
        base_path = os.path.join(self.skill_dir, "base.md")
        if os.path.exists(base_path):
            content = self._read_file_safe(base_path)
            if content:
                combined_content.append(f"### 核心记忆: base.md ###\n{content}\n")

        # 2. 加载 catalogue.md (动态过滤禁用项)
        cat_path = os.path.join(self.skill_dir, "catalogue.md")
        if os.path.exists(cat_path):
            cat_content = self._read_file_safe(cat_path)
            if cat_content:
                filtered_lines = []
                disabled_skills = self.config.config.get("disabled_skills", [])
                
                for line in cat_content.split('\n'):
                    # 如果该行包含任何被禁用的技能名称，则直接跳过该行
                    if any(disabled in line for disabled in disabled_skills):
                        continue
                    filtered_lines.append(line)
                
                filtered_cat = "\n".join(filtered_lines)
                combined_content.append(f"### 技能目录: catalogue.md ###\n{filtered_cat}\n")
        
        logger.info(f">>> [SYSTEM] 成功加载核心记忆与已启用的目录")
        return "\n".join(combined_content) if combined_content else "【暂无核心记忆或目录】"

    def _load_files_from_dir(self, directory, label):
        combined_content = []
        if not os.path.exists(directory):
            os.makedirs(directory)
            return f"【{label}库目前为空】"

        files = [f for f in os.listdir(directory) if f.endswith(".md")]
        if not files:
            return f"【{label}库暂无内容】"

        for filename in files:
            file_path = os.path.join(directory, filename)
            file_content = self._read_file_safe(file_path)
            if file_content:
                combined_content.append(f"### {label}来源: {filename} ###\n{file_content}\n")
        
        return "\n".join(combined_content)

    def load_all_skills(self):
        return self._load_files_from_dir(self.skill_dir, "技能笔记")

    def load_allowlist(self):
        """加载前检查配置开关"""
        if not self.config.is_allowlist_enabled():
            return "【白名单机制当前已在配置中禁用。遇到任何敏感操作请直接申请审批。】"
        return self._load_files_from_dir(self.allowlist_dir, "白名单记录")