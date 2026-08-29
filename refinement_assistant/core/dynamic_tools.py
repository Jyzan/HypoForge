# core/dynamic_tools.py
import os
from tools.registry import TOOL_REGISTRY
from logger_manager import logger

class DynamicToolManager:
    def __init__(self, workspace_path, config):
        self.workspace_path = workspace_path
        self.config = config # 注入配置管理器
        self.active_tools = []
        self.meta_funcs = {}
        self.tool_definitions = []
        self._setup_meta_tools()
        self._refresh_tool_definitions()

    def reset(self):
        self.active_tools = []
        self._refresh_tool_definitions()

    def get_tool_catalogue_text(self):
        """【新增】动态生成说明书，直接剔除被禁用的工具"""
        desc = []
        for name, info in TOOL_REGISTRY.items():
            if self.config.is_tool_enabled(name):
                # 只将启用的工具加入说明书
                desc.append(f"- {name}: {info['summary']}")
        return "\n".join(desc)

    def _setup_meta_tools(self):
        def read_skill_content(skill_name):
            # 防止 AI 强制读取被禁用的技能文件
            if not self.config.is_skill_enabled(skill_name):
                return f"❌ 权限拒绝：技能 '{skill_name}' 已被管理员禁用。"
                
            skill_path = os.path.join(self.workspace_path, "skills", skill_name)
            encodings = ['utf-8', 'utf-8-sig', 'utf-16', 'gbk']
            for enc in encodings:
                try:
                    with open(skill_path, 'r', encoding=enc) as f:
                        return f.read()
                except:
                    continue
            return f"❌ 文件 {skill_name} 不存在或读取失败。"

        def load_tool_definition(tool_names):
            if isinstance(tool_names, str):
                tool_names = [tool_names]
                
            results = []
            changed = False
            for tool_name in tool_names:
                # 【硬拦截】尝试挂载被禁用的工具直接报错
                if not self.config.is_tool_enabled(tool_name):
                    results.append(f"❌ 权限拒绝：工具 '{tool_name}' 已被管理员禁用，无法挂载。")
                    continue
                    
                if tool_name in TOOL_REGISTRY:
                    if tool_name not in self.active_tools:
                        self.active_tools.append(tool_name)
                        results.append(f"✅ 工具 '{tool_name}' 已成功挂载。")
                        changed = True
                    else:
                        results.append(f"ℹ️ 工具 '{tool_name}' 已经在运行列表中。")
                else:
                    results.append(f"❌ 未找到工具 '{tool_name}'，请检查工具说明书。")
            
            if changed:
                self._refresh_tool_definitions()
                
            return "\n".join(results)

        self.meta_funcs = {
            "read_skill": read_skill_content,
            "load_tool": load_tool_definition,
            "delegate_task": lambda **kwargs: "此任务由 main.py 拦截处理" # 仅占位，实际在 main 拦截
        }

    def _refresh_tool_definitions(self):
        new_defs = []
        new_defs.append({
            "type": "function",
            "function": {
                "name": "read_skill",
                "description": "读取 skills/ 目录下的特定技能文件详细内容。",
                "parameters": {
                    "type": "object",
                    "properties": {"skill_name": {"type": "string"}},
                    "required": ["skill_name"]
                }
            }
        })
        new_defs.append({
            "type": "function",
            "function": {
                "name": "load_tool",
                "description": "动态挂载工具的完整能力。支持一次性批量挂载多个工具。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tool_names": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["tool_names"]
                }
            }
        })
        new_defs.append({
            "type": "function",
            "function": {
                "name": "delegate_task",
                "description": "【核心指令】将底层操作委派给 Worker 执行。收到战报后，立刻根据战报直接回复用户，终止思考。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_description": {
                            "type": "string", 
                            "description": "给 Worker 的极度详尽的SOP。必须在文本中用方括号打上工具标签，如：'使用 [run_powershell] 列出目录'。不打标签会导致系统不给士兵分配工具而失败！"
                        }
                    },
                    "required": ["task_description"]
                }
            }
        })
        
        for name in self.active_tools:
            if name in TOOL_REGISTRY and self.config.is_tool_enabled(name):
                new_defs.append(TOOL_REGISTRY[name]["definition"])
        
        self.tool_definitions = new_defs
        logger.info(f"[TOOLS] 当前挂载工具数: {len(self.tool_definitions)}")

    def is_meta_tool(self, name):
        return name in self.meta_funcs

    def is_active_tool(self, name):
        return name in TOOL_REGISTRY and name in self.active_tools and self.config.is_tool_enabled(name)

    def execute_meta(self, name, args):
        return self.meta_funcs[name](**args)
        
    def get_definitions(self):
        return self.tool_definitions