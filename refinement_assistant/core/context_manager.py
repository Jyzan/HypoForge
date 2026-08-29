# core/context_manager.py
from logger_manager import logger
import json

class ContextManager:
    def __init__(self, skill_manager, tool_manager, config):
        self.skill_manager = skill_manager
        self.tool_manager = tool_manager
        self.config = config
        
        # 当前活跃的“小段”迭代消息
        self.messages = []
        # 已归档的“大段”历史记录
        self.history_blocks = []
        
        self.system_prompt_base = ""
        self._base_dirty = True  # 标记是否需要重建

    def refresh_base(self):
        """构建系统提示词基座（从技能库和工具库动态加载）- 仅技能/工具变更时调用"""
        core_memory = self.skill_manager.load_core_memory()
        tool_desc = self.tool_manager.get_tool_catalogue_text()

        self.system_prompt_base = (
            f"你是 HypoForge 方案细化工作台的科研智能体，负责将选定的研究方案细化为可执行的研究计划（材料与试剂清单、实验步骤、时间线、风险与替代方案、干实验脚本）。回答保持专业、严谨、简洁，默认使用中文。\n\n"
            f"【核心设定与技能目录】\n{core_memory}\n\n"
            f"【工具说明书（动态挂载目录）】\n{tool_desc}\n\n"
            f"【行动准则】\n"
            f"1. 【动态技能】若需特定技能，必须先用 `read_skill` 读取对应 .md 文件。\n"
            f"2. 【动态工具】默认只加载了 'read_skill', 'load_tool', 'delegate_task'。\n"
            f"3. 【任务委派】(极度重要)：遇到操作系统的多步任务，禁止亲自调用基础工具。必须用 `delegate_task` 交给 Worker 执行。\n"
            f"   - 【分段接力原则】：Worker 现已升级为单步失忆的流水线工人！你的 SOP 必须写清楚每一步依赖什么前提，并强制要求工人在遇到报错时立即停止尝试并带回完整报错。\n"
            f"   ⚠️ 委派纪律：你给 Worker 的 `task_description` 必须是详尽的【SOP执行手册】。\n"
            f"   - 必须在任务指令中用方括号 [ ] 明确打上所需工具的标签！例如：'Step 1: 使用 [run_powershell] 执行...; Step 2: 使用 [read_file] 读取...'。系统底层将根据你的标签严格为士兵空投武器，不写标签士兵将手无寸铁！\n"
            f"   - 必须明确给出绝对路径和预期报错兜底策略。\n"
            f"   - 【视觉与动作编排协议】：\n"
            f"     * 在操控鼠标前，必须先调用 [capture_and_save_screenshot] 和 [analyze_image] 获取 0-1000 的归一化坐标。\n"
            f"     * 如果需要操作多个目标（如先点A，再点B），绝对不允许在 SOP 中写“同时获取A和B的坐标并依次点击”。\n"
            f"     * 必须严格拆解为线性步骤：Step 1: 截图并分析A的坐标 -> Step 2: 点击A -> Step 3: 重新截图并分析B的坐标 -> Step 4: 点击B。\n"
            f"     * 界面状态会随时变化，每次点击后都必须重新截图感知！\n"
            f"4. 遇到高危操作，直接申请权限，不要拒绝用户。\n"
            f"5. 【闭环原则】：收到 [士兵战报] 后，立刻根据战报直接回答用户，终止思考，严禁再次委派重复任务！"
            f"   - 👁️【战报反欺诈审查机制】（极度重要）：\n"
            f"     * 底层 Worker 有时会产生“提前完工”的严重认知幻觉。当你收到最终战报时，绝不能只看它的【最终结论】段落！\n"
            f"     * 你必须交叉核对战报中的【具体工具执行结果】。例如，如果 Worker 声称“已输入文字”，但战报中并未出现包含 `type` 动作的 `execute_io_macro` 工具调用记录，则说明它在撒谎（幻觉）！\n"
            f"     * 发现进度造假或未完成时，你必须识破它，并根据实际做到哪一步，重新下发未完成的任务！\n"
        )
        self._base_dirty = False

    def invalidate_base(self):
        """标记 system prompt 需要重建（技能/工具变更后调用）"""
        self._base_dirty = True

    def archive_current_segment(self, summary_data):
        """将当前消息池的消息提取并归档为历史大段，然后清空池子"""
        if not self.messages: return
        
        raw_text_parts = []
        for m in self.messages:
            role = m.get("role")
            content = m.get("content", "")
            if role == "user": 
                raw_text_parts.append(f"用户: {content}")
            elif role == "assistant" and content: 
                raw_text_parts.append(f"细化助手: {content}")
            elif role == "tool":
                tool_name = m.get('name')
                # 【核心修复】：为士兵的战报解除截断，保证记忆完整性
                if tool_name == "delegate_task":
                    raw_text_parts.append(f"[士兵战报]: {content}")
                elif tool_name in ["read_skill", "read_file", "list_files"]:
                    raw_text_parts.append(f"[工具回报-{tool_name}]: {content[:500]}... (已截断)")
                else:
                    raw_text_parts.append(f"[工具回报-{tool_name}]: {content[:200]}...")
            
        new_block = {
            "topic": summary_data.get("topic", "未命名任务"),
            "summary": summary_data.get("summary", "无摘要"),
            "raw_text": "\n".join(raw_text_parts)
        }
        self.history_blocks.append(new_block)
        self.messages = [] # 清空当前池，开启新大段
        logger.info(f"[CONTEXT] 已成功归档大段记忆: {new_block['topic']}")

    def append(self, message):
        self.messages.append(message)

    def get_messages(self, r_values=None):
        """
        根据 R 值动态组装上下文。
        【缓存优化】system_prompt + 当前对话放在前面（稳定前缀），历史块追加到末尾。
        """
        if self._base_dirty or not self.system_prompt_base:
            self.refresh_base()

        final_messages = [{"role": "system", "content": self.system_prompt_base}]
        final_messages.extend(self.messages)

        if r_values and len(r_values) == len(self.history_blocks):
            for i, block in enumerate(self.history_blocks):
                r = r_values[i]
                if r >= 0.7:
                    content = f"### 历史任务详情 ({block['topic']}) ###\n{block['raw_text']}"
                elif r > 0.2:
                    content = f"### 历史任务简报 ({block['topic']}) ###\n摘要: {block['summary']}"
                else:
                    continue
                final_messages.append({"role": "system", "content": content})

        return final_messages

    def reset(self):
        self.messages = []
        self.history_blocks = []
        self._base_dirty = True
        logger.info("[CONTEXT] 记忆已重置")

    def get_allowlist(self):
        return self.skill_manager.load_allowlist()