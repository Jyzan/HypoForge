# main.py
import os, json
from pathlib import Path
import html
import threading
import time
from tools.registry import TOOL_REGISTRY 
from token_manager import TokenManager 
from logger_manager import logger 
from learning_skills import SkillManager
from subagent.security_agent import SecurityAgent
from subagent.organizer_agent import OrganizerAgent
from subagent.summarizer_agent import SummarizerAgent
from subagent.worker_agent import WorkerAgent

from config import ConfigManager
from core.dynamic_tools import DynamicToolManager
from core.context_manager import ContextManager
from core.llm_client import create_llm_client, get_chat_model, message_to_dict
from core.permission_manager import PermissionManager

class AIAssistant:
    def __init__(self):
        # 基础组件初始化
        self.client = create_llm_client()
        self.endpoint_id = get_chat_model()
        self._stop_checker = None  # 外部停止信号检查函数
        self.workspace_path = os.getenv("WORKSPACE") or str(Path(__file__).resolve().parent)

        # 0. 配置中心需先于 TokenManager，保证持久化的 token_limit 生效
        self.config_manager = ConfigManager(self.workspace_path)
        self.token_manager = TokenManager(
            default_limit=self.config_manager.config.get("token_limit")
        )
        self.pending_approvals = {}
        self.permission_manager = PermissionManager(self.workspace_path, self.config_manager.config)
        
        # 2. 挂载子代理 (Sub-Agents)
        self.skill_manager = SkillManager(self.workspace_path, self.config_manager)
        self.security_agent = SecurityAgent(self.client, self.endpoint_id, self.token_manager)
        self.organizer_agent = OrganizerAgent(self.client, self.endpoint_id, self.token_manager, self.workspace_path)
        self.summarizer_agent = SummarizerAgent(self.client, self.endpoint_id, self.token_manager)
        self.worker_agent = WorkerAgent(self.client, self.endpoint_id, self.token_manager, self.security_agent, self.pending_approvals, self.permission_manager)
        
        # 3. 核心逻辑模块 (Core Modules)
        self.tool_manager = DynamicToolManager(self.workspace_path, self.config_manager)
        self.context_manager = ContextManager(self.skill_manager, self.tool_manager, self.config_manager)

        # 存储当前 R 值分布
        self.current_r_values = []
        # 当前持久化会话 ID（用于历史对话的增量保存）
        self.current_session_id = None
        # 当前已载入固定方案的 run_id（避免同一方案重复 kickoff），从持久化配置恢复
        self.current_refinement_run_id = self.config_manager.config.get("last_refinement_run_id") or None

    def reset_chat(self):
        """完全重置会话，清空记忆与 Token 计数"""
        self.context_manager.reset()
        self.tool_manager.reset()
        self.token_manager.reset_usage()
        self.current_r_values = []
        self.current_session_id = None
        self.current_refinement_run_id = None
        self._stop_checker = None
        logger.info("\n" + "="*80 + "\n[RESET] 开启全新篇章" + "\n" + "="*80)

    def load_session(self, data):
        """从持久化会话恢复完整上下文（消息 + 归档记忆），并继续写回同一会话文件"""
        self.context_manager.reset()
        self.tool_manager.reset()
        # Token 额度按对话隔离：恢复该会话自己的累计用量
        self.token_manager.restore_usage(data.get("tokens_used"))
        self.current_r_values = []
        self.context_manager.messages = [
            {"role": m.get("role"), "content": m.get("content", "")}
            for m in data.get("messages", [])
            if m.get("role") in ["user", "assistant"] and m.get("content")
        ]
        blocks = data.get("history_blocks")
        if isinstance(blocks, list):
            self.context_manager.history_blocks = blocks
        self.current_session_id = data.get("id")
        self._stop_checker = None
        logger.info(
            f"[SESSION] 已加载会话 {self.current_session_id} "
            f"({len(self.context_manager.messages)} 条消息, {len(self.context_manager.history_blocks)} 个归档块)"
        )

    def set_stop_checker(self, checker):
        """设置外部停止信号检查函数。checker() 返回 True 时应立即停止生成。"""
        self._stop_checker = checker

    def boost_token_limit(self):
        """临时翻倍 Token 阈值"""
        self.token_manager.double_threshold()
        
    def set_token_limit(self, new_limit):
        """自定义 Token 阈值，并持久化到 config.json"""
        self.token_manager.set_threshold(new_limit)
        self.config_manager.config["token_limit"] = self.token_manager.max_threshold
        self.config_manager.save()

    def resume_archive(self, messages):
        """从消息存档恢复当前对话上下文。"""
        self.context_manager.reset()
        self.tool_manager.reset()
        self.token_manager.reset_usage()
        self.current_r_values = []
        self.context_manager.messages = [
            {"role": m.get("role"), "content": m.get("content", "")}
            for m in messages
            if m.get("role") in ["user", "assistant"] and m.get("content")
        ]
        logger.info(f"[ARCHIVE] 已从消息存档恢复 {len(self.context_manager.messages)} 条消息")

    def _estimate_message_tokens(self):
        """粗略估算当前消息池的 token 数 (中文约 1.5 字符/token，英文约 4 字符/token)"""
        total_chars = 0
        for m in self.context_manager.messages:
            content = m.get("content", "") or ""
            total_chars += len(content)
        return int(total_chars / 2.5)  # 混合中英文的粗略估算

    def _auto_compact_if_needed(self, force=False):
        """Claude Code 风格自动压缩：按"上下文窗口压力"触发——最近一次请求的
        prompt tokens（字符估算兜底）接近模型上下文窗口时压缩，而不是按累计
        消耗预算（额度是花费概念，上下文窗口才是溢出概念）。
        保留最近 3 轮完整对话，旧内容用 LLM 摘要替代。force=True 用于溢出报错后强制重试。
        """
        min_messages = 8  # 至少 4 轮对话才压缩
        if len(self.context_manager.messages) < min_messages and not force:
            return False

        # 真实上下文占用优先（API 返回的 prompt_tokens），字符估算兜底
        effective_tokens = max(self.token_manager.last_prompt_tokens, self._estimate_message_tokens())
        threshold = int(self.token_manager.context_window * 0.7)
        if not force and effective_tokens < threshold:
            return False

        logger.info(
            f"[AUTO-COMPACT] 触发压缩！上下文占用约 {effective_tokens}/{self.token_manager.context_window} "
            f"(>= {threshold}), 消息数 {len(self.context_manager.messages)}"
        )

        # 保留最后 6 条消息（最近 3 轮 user/assistant）
        keep_count = 6
        to_keep = self.context_manager.messages[-keep_count:]
        to_compress = self.context_manager.messages[:-keep_count]

        if len(to_compress) < 4:
            if not force or not to_compress:
                return False

        # 调用 LLM 生成摘要
        summary = self._summarize_messages_for_compact(to_compress)
        if not summary:
            logger.warning("[AUTO-COMPACT] 摘要生成失败，跳过压缩")
            return False

        # 重建消息池：摘要 + 最近 3 轮
        self.context_manager.messages = [
            {"role": "system", "content": summary},
            *to_keep,
        ]

        # 重要：不重置 usage，因为压缩不影响已消耗的 token
        # 但后续每轮请求的 token 数会因上下文变小而显著降低
        compressed = len(to_compress)
        kept = len(to_keep)
        logger.info(
            f"[AUTO-COMPACT] 完成！压缩 {compressed} 条 → 摘要，保留最近 {kept} 条。"
            f"摘要长度: {len(summary)} 字符"
        )
        return True

    @staticmethod
    def _is_context_overflow(err):
        """识别各平台的上下文超长报错"""
        text = str(err).lower()
        keywords = (
            "maximum context length", "context length", "input length exceed",
            "range of input length", "prompt is too long", "too many tokens",
            "exceed context limit",
        )
        return any(k in text for k in keywords)

    def _summarize_messages_for_compact(self, messages):
        """用 LLM 将旧消息压缩为结构化摘要（仿 Claude Code compact）。"""
        # 提取对话文本
        conversation_text = ""
        for m in messages:
            role = m.get("role", "?")
            content = m.get("content", "")
            if role == "system":
                continue  # 旧的系统提示不再需要
            if role == "user":
                conversation_text += f"用户: {content}\n"
            elif role == "assistant":
                # 截断 AI 回复的冗余部分
                if content and len(content) > 300:
                    content = content[:300] + "..."
                conversation_text += f"AI: {content}\n"
            elif role == "tool":
                name = m.get("name", "?")
                short = content[:200] + "..." if len(content) > 200 else content
                conversation_text += f"[工具 {name} 结果]: {short}\n"

        if len(conversation_text) < 200:
            return None

        prompt = (
            "请将以下对话历史压缩为一段结构化摘要。只输出摘要，不要任何前缀或解释。\n\n"
            "要求：\n"
            "1. 保留所有用户明确提出的任务和需求\n"
            "2. 记录已完成的操作和结果\n"
            "3. 记录进行中/未完成的操作\n"
            "4. 记录关键决策和发现\n"
            "5. 丢弃工具调用的冗余细节（坐标、截图路径、耗时等）\n"
            "6. 中文输出，简洁但信息完整\n\n"
            "格式参考：\n"
            "## 对话摘要\n"
            "- 用户任务: ...\n"
            "- 已完成: ...\n"
            "- 进行中: ...\n"
            "- 关键发现: ...\n\n"
            f"对话历史:\n{conversation_text}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.endpoint_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0,
            )
            summary = response.choices[0].message.content or ""
            summary = summary.strip()
            # 记录摘要消耗
            if hasattr(response, 'usage'):
                self.token_manager.update_usage(response, source="AutoCompact")
            return summary if len(summary) > 20 else None
        except Exception as e:
            logger.error(f"[AUTO-COMPACT] LLM 摘要失败: {e}")
            return None

    def chat_yield(self, user_input=None):
        """对话入口：将消息累积形成稳定前缀以利用 DeepSeek 磁盘缓存"""
        if user_input:
            logger.info(f"【用户指令】: {user_input}")

            # 仅在消息池过大时归档旧消息，避免破坏缓存前缀
            # （按上下文窗口判断，字符估算兜底；窗口的一半即应归档）
            est_tokens = self._estimate_message_tokens()
            if est_tokens > self.token_manager.context_window * 0.5:
                yield json.dumps({"type": "status", "content": "正在压缩历史对话以释放上下文..."})
                summary_data = self.summarizer_agent.summarize(self.context_manager.messages)
                self.context_manager.archive_current_segment(summary_data)

            if self.context_manager.history_blocks:
                yield json.dumps({"type": "status", "content": "正在检索相关历史 (R值)..."})
                self.current_r_values = self.summarizer_agent.rank_relevance(
                    self.context_manager.history_blocks, user_input
                )
                if len(self.current_r_values) >= 1:
                    self.current_r_values[-1] = round(min(1.0, self.current_r_values[-1] + 0.4), 2)
                if len(self.current_r_values) >= 2:
                    self.current_r_values[-2] = round(min(1.0, self.current_r_values[-2] + 0.2), 2)
                logger.info(f"[RANK] 提权后 R值分布: {self.current_r_values}")

            self.context_manager.append({"role": "user", "content": user_input})

        gen = self._main_loop()
        for response in gen:
            yield response

    def organize_yield(self, item, details=""):
        """处理 /organize 指令的专属流"""
        logger.info(f"【指令请求】: /organize {item} | 细节: {details}")
        yield json.dumps({"type": "status", "content": "🧹 Organizer 子代理已启动..."})
        
        # 1. 调用 Agent 获取生成器
        result_gen = self.organizer_agent.organize(item, details)
        
        # 2. 核心修复：透明转发 Agent 产生的内容
        # 既然 Agent 内部已经 yield 了 JSON 字符串，这里直接转发
        try:
            yield from result_gen
            
            # 3. 整理成功后的后续处理
            yield json.dumps({"type": "status", "content": "🔄 文件整理完毕，正在重载目录..."})
            self.context_manager.refresh_base()
            
        except Exception as e:
            logger.error(f"整理流处理异常: {e}")
            yield json.dumps({
                "type": "final", 
                "content": f"❌ 整理过程发生系统错误: {str(e)}", 
                "tokens": self.token_manager.get_status_report()
            })

    def memory_yield(self, mode, name):
        """处理 /memory [mode] [name] 指令"""
        import os, json
        # 创建一个专门存放记忆快照的目录
        memories_dir = os.path.join(self.workspace_path, "memories")
        if not os.path.exists(memories_dir):
            os.makedirs(memories_dir)
            
        file_path = os.path.join(memories_dir, f"{name}.md")
        
        if mode == "save":
            # 1. 保存前，必须把当前正在进行的小段对话也压缩打包进去
            if self.context_manager.messages:
                yield json.dumps({"type": "status", "content": "正在压缩当前尚未归档的对话碎片..."})
                summary_data = self.summarizer_agent.summarize(self.context_manager.messages)
                self.context_manager.archive_current_segment(summary_data)
                
            # 2. 将整个历史大段导出
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    # 以 JSON 格式写入 .md 文件，因为包含摘要和原文，JSON 格式读取最稳定
                    f.write(json.dumps(self.context_manager.history_blocks, ensure_ascii=False, indent=2))
                
                yield json.dumps({
                    "type": "final", 
                    "content": f"💾 **记忆快照已保存**\n当前的对话段落已成功封存至 `memories/{name}.md`。",
                    "tokens": self.token_manager.get_status_report()
                })
            except Exception as e:
                logger.error(f"[MEMORY] 保存失败: {e}")
                yield json.dumps({"type": "error", "content": f"保存记忆失败: {e}"})
                
        elif mode == "load":
            if not os.path.exists(file_path):
                yield json.dumps({"type": "error", "content": f"❌ 未找到记忆文件: {name}.md"})
                return
                
            yield json.dumps({"type": "status", "content": f"正在读取记忆碎片 {name}.md ..."})
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    loaded_blocks = json.loads(f.read())
                    
                # 核心巧妙点：直接推入 history_blocks 数组中
                # 下一次发指令时，SummarizerAgent 就会对它们进行 R值 计算！
                self.context_manager.history_blocks.extend(loaded_blocks)
                
                yield json.dumps({
                    "type": "final", 
                    "content": f"📂 **记忆已挂载**\n成功读取并注入 {len(loaded_blocks)} 个记忆大段！\n这些记忆将在你的下一次对话中接受 **R值相关性检索** 的调度。这不会立刻增加你的 Token 负担。",
                    "tokens": self.token_manager.get_status_report()
                })
            except Exception as e:
                logger.error(f"[MEMORY] 读取失败: {e}")
                yield json.dumps({"type": "error", "content": f"读取记忆失败，文件可能被篡改导致格式损坏: {e}"})
        else:
            yield json.dumps({"type": "error", "content": f"未知的操作模式: {mode}"})

    def _main_loop(self):
        """核心 LLM 迭代循环"""
        max_turns = 12
        turn = 0
        try:
            while turn < max_turns:
                turn += 1
                # 检查外部停止信号
                if self._stop_checker and self._stop_checker():
                    yield json.dumps({"type": "error", "content": "用户请求停止生成"})
                    return
                if self.token_manager.is_over_limit():
                    logger.warning("[SYSTEM] Token 达到阈值，停止迭代")
                    yield json.dumps({
                        "type": "final",
                        "content": "Token 额度已耗尽。可在「设置」中调大上限，或使用 /new 开启新对话。",
                        "tokens": self.token_manager.get_status_report()
                    })
                    return

                yield json.dumps({"type": "status", "content": "正在思考..."})

                # Claude Code 风格自动压缩：上下文窗口压力达到 70% 时触发
                if self._auto_compact_if_needed():
                    yield json.dumps({"type": "status", "content": "对话历史过长，正在自动压缩..."})

                messages = self.context_manager.get_messages(self.current_r_values)

                try:
                    response = self.client.chat.completions.create(
                        model=self.endpoint_id,
                        messages=messages,
                        tools=self.tool_manager.get_definitions(),
                        tool_choice="auto"
                    )
                except Exception as api_err:
                    # 上下文溢出兜底：估算不及时时 API 会直接报错，强制压缩后重试一次
                    if not self._is_context_overflow(api_err):
                        raise
                    logger.warning(f"[CONTEXT OVERFLOW] 上下文超窗，强制压缩后重试: {api_err}")
                    yield json.dumps({"type": "status", "content": "上下文超出模型窗口，正在强制压缩后重试..."})
                    if not self._auto_compact_if_needed(force=True):
                        raise
                    messages = self.context_manager.get_messages(self.current_r_values)
                    response = self.client.chat.completions.create(
                        model=self.endpoint_id,
                        messages=messages,
                        tools=self.tool_manager.get_definitions(),
                        tool_choice="auto"
                    )
                self.token_manager.update_usage(response, source="General-MainLoop")
                msg = response.choices[0].message

                # 推理模型 fallback：content 为空时用 reasoning_content
                content = getattr(msg, 'content', None) or ""
                if not content:
                    reasoning = getattr(msg, 'reasoning_content', None)
                    if reasoning:
                        logger.warning(f"[REASONING FALLBACK] content 为空，用 reasoning_content 作为回复 ({len(reasoning)} 字符)")
                        content = reasoning
                    else:
                        logger.error("[REASONING FALLBACK] content 和 reasoning_content 均为空！可能是模型内部错误")

                # 记录输出摘要到日志
                if content:
                    log_text = content.replace('\n', ' ')[:100]
                    logger.info(f"[AI RESPONSE] {log_text}...")

                self.context_manager.append(message_to_dict(msg))

                if getattr(msg, 'tool_calls', None):
                    for tool_call in msg.tool_calls:
                        logger.info(f"[GENERAL TOOL CALL] ID: {tool_call.id} | Function: {tool_call.function.name} | 参数: {tool_call.function.arguments}")
                        yield from self._run_tool_logic(tool_call)
                    continue # 继续下一轮迭代处理工具结果
                else:
                    yield json.dumps({"type": "final", "content": content or "[DeepSeek 返回了空内容，请重试]", "tokens": self.token_manager.get_status_report()})
                    return
        except Exception as e:
            logger.error(f"[CORE ERROR] 运行时异常: {e}", exc_info=True)
            yield json.dumps({"type": "error", "content": str(e)})

    def _run_tool_logic(self, tool_call):
        """工具分发与安全审计"""
        # =========================================================
        # 【核心修复】：将导入语句提升到函数最顶端，变为全局可用
        # =========================================================
        from tools.registry import TOOL_REGISTRY
        
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        if name == "run_powershell" and isinstance(args.get("command"), str):
            args["command"] = html.unescape(args["command"])
        
        logger.info(f"[GENERAL TOOL CALL] ID: {tool_call.id} | Function: {name} | 参数: {tool_call.function.arguments}")
        
        if name == "delegate_task":
            task_desc = args.get("task_description", "")
            yield json.dumps({
                "type": "mobile_progress",
                "content": f"【委派任务】\n{task_desc}",
            })
            yield json.dumps({"type": "status", "content": "已将任务委派给执行子代理..."})
            
            import re
            tagged_tools = re.findall(r'\[([a-zA-Z0-9_]+)\]', task_desc)
            logger.info(f"[SYSTEM] 侦测到将军打的工具标签: {tagged_tools}")
            
            worker_tools = []
            worker_active = []
            
            # 这里的 from tools.registry import TOOL_REGISTRY 已经被移走了！
            for t_name, info in TOOL_REGISTRY.items():
                if self.config_manager.is_tool_enabled(t_name):
                    if not tagged_tools or t_name in tagged_tools:
                        worker_tools.append(info["definition"])
                        worker_active.append(t_name)
            
            logger.info(f"[SYSTEM] 实际为空投士兵分配了 {len(worker_tools)} 个工具。")
            allowlist = self.context_manager.get_allowlist()
            
            worker_result = yield from self.worker_agent.execute_task(task_desc, worker_tools, worker_active, allowlist)

            if hasattr(worker_result, "summary_for_model"):
                res_content = worker_result.summary_for_model
                mobile_feedback = worker_result.final_feedback or worker_result.summary_for_model
            else:
                res_content = str(worker_result)
                mobile_feedback = res_content
            if "### 最终结论 ###" in mobile_feedback:
                mobile_feedback = mobile_feedback.rsplit("### 最终结论 ###", 1)[-1].strip()
            yield json.dumps({
                "type": "mobile_progress",
                "content": f"【执行反馈】\n{mobile_feedback}",
            })
            yield json.dumps({"type": "status", "content": "子代理执行完成，正在汇总结论..."})

        # ... (下方处理 read_skill 和 run_powershell 等逻辑保持不变) ...
        # 1. 处理元工具 (Meta Tools: read_skill, load_tool)
        elif self.tool_manager.is_meta_tool(name):
            res_content = str(self.tool_manager.execute_meta(name, args))
            yield json.dumps({"type": "status", "content": f"⚡ 元工具 {name} 执行成功"})
            
        # 2. 处理已挂载的活跃工具
        elif self.tool_manager.is_active_tool(name):
            res_content = None
            permission_gate_approved = False
            permission_decision = self.permission_manager.evaluate(name, args)
            if permission_decision.action == "deny":
                res_content = f"权限拒绝: {permission_decision.reason} | {permission_decision.message}"
                yield json.dumps({"type": "status", "content": res_content})
            else:
                while permission_decision.action == "ask":
                    approved = yield from self._await_tool_approval(
                        tool_call.id,
                        {
                            "type": "approval",
                            "tool_name": name,
                            "command": permission_decision.command or json.dumps(args, ensure_ascii=False),
                            "tool_call_id": tool_call.id,
                            "audit_type": permission_decision.audit_type,
                            "sandbox_path": permission_decision.sandbox_path,
                            "message": permission_decision.message,
                        },
                        "等待用户确认权限边界...",
                    )
                    if not approved:
                        res_content = f"权限受限: 用户未批准 {permission_decision.reason}"
                        break
                    permission_gate_approved = True
                    if permission_decision.audit_type != "SANDBOX":
                        break
                    permission_decision = self.permission_manager.evaluate(name, args)
                    if permission_decision.action == "deny":
                        res_content = (
                            f"权限拒绝: {permission_decision.reason} | "
                            f"{permission_decision.message}"
                        )
                        break

                if permission_decision.rewritten_args and permission_gate_approved:
                    args.update(permission_decision.rewritten_args)

            if (
                permission_decision.action != "deny"
                and res_content is None
                and name == "run_powershell"
            ):
                cmd = args.get("command", "")
                audit_res = (
                    "SAFE"
                    if permission_gate_approved
                    else self.security_agent.audit(
                        cmd,
                        self.context_manager.get_allowlist(),
                    )
                )
                
                if audit_res in ["WARNING", "DANGER"]:
                    logger.warning(f"[SECURITY] 触发安全拦截: {audit_res} | 指令: {cmd}")
                    approved = yield from self._await_tool_approval(
                        tool_call.id,
                        {
                            "type": "approval",
                            "tool_name": name,
                            "command": cmd,
                            "tool_call_id": tool_call.id,
                            "audit_type": audit_res,
                        },
                        "危险操作，正在等待用户审批...",
                    )
                    if not approved:
                        res_content = "权限受限: 用户未批准危险操作"

                if res_content is None:
                    res_content = ""
                    for line in TOOL_REGISTRY[name]["function"](**args):
                        res_content += line + "\n"
                        # UI 显示限制，防止长输出卡死前端
                        display_line = line[:60] + "..." if len(line) > 60 else line
                        yield json.dumps({"type": "status", "content": f"> {display_line}"})
            elif permission_decision.action != "deny" and res_content is None:
                # 普通工具执行
                func = TOOL_REGISTRY[name]["function"]
                res_content = str(func(**args))
                yield json.dumps({"type": "status", "content": f"⚡ 工具 {name} 完成"})
        else:
            res_content = f"❌ 执行失败：工具 '{name}' 未就绪或已被禁用。"

        self.context_manager.append({
            "tool_call_id": tool_call.id, 
            "role": "tool", 
            "name": name, 
            "content": res_content
        })

    def _await_tool_approval(self, tool_call_id, payload, wait_message):
        event = threading.Event()
        self.pending_approvals[tool_call_id] = {
            "event": event,
            "approved": False,
        }
        yield json.dumps(payload)
        yield json.dumps({"type": "status", "content": wait_message})
        self._wait_for_approval_event(event, timeout=300)
        approval_data = self.pending_approvals.pop(tool_call_id, None)
        return bool(approval_data and approval_data.get("approved"))

    @staticmethod
    def _wait_for_approval_event(event, timeout=300, poll_interval=0.25):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if event.wait(timeout=min(poll_interval, max(0, deadline - time.monotonic()))):
                return True
            try:
                from tools import powershell_ops
                if powershell_ops.is_cancelled():
                    return False
            except Exception:
                pass
        return event.is_set()

    def handle_approval(
        self,
        tool_call_id,
        command,
        is_approved,
        audit_type=None,
        persist_warning=True,
    ):
        """【核心重构】不再亲自执行工具，也不再抛出迭代器！仅仅唤醒挂起的线程"""
        if is_approved and audit_type == "SANDBOX":
            sandbox_path = os.path.abspath(command or os.path.join(self.workspace_path, "sandbox"))
            os.makedirs(sandbox_path, exist_ok=True)
            self.config_manager.config["sandbox_confirmed"] = True
            self.config_manager.config["sandbox_path"] = sandbox_path
            self.config_manager.save()
            if hasattr(self, "permission_manager"):
                self.permission_manager.refresh(self.config_manager.config)

        if tool_call_id in self.pending_approvals:
            logger.info(f"[APPROVAL] 指挥官下达审批结果: {'批准' if is_approved else '拒绝'} | 指令: {command}")
            
            # 若批准且是 WARNING，加入白名单
            if is_approved and audit_type == "WARNING" and persist_warning:
                self._add_to_allowlist(command)
            
            # 修改状态并唤醒还在 _run_worker_tool 里死等的士兵线程！
            self.pending_approvals[tool_call_id]["approved"] = is_approved
            self.pending_approvals[tool_call_id]["event"].set()
        else:
            logger.warning(f"[APPROVAL ERROR] 未找到对应的待审批挂起任务: {tool_call_id}")
            
        return ""

    def parse_approval_intent(self, text, approval_context):
        """Parse a free-form mobile approval into a constrained JSON intent."""
        prompt = (
            "将用户对权限请求的回复解析为 JSON。只输出一个 JSON 对象，不要 Markdown。\n"
            "允许字段严格为：decision, scope, capabilities, confident。\n"
            "decision 只能是 approve、deny、revoke；scope 只能是 "
            "operation、task、conversation；capabilities 只能从 "
            "all、non_command、sandbox_write、uv_command 中选择。\n"
            "无法确定时 confident=false，且不要扩大授权。\n"
            "critical=true 时只能使用 operation 范围。\n"
            f"权限上下文：{json.dumps(approval_context, ensure_ascii=False)}\n"
            f"用户回复：{text}"
        )
        response = self.client.chat.completions.create(
            model=self.endpoint_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        content = str(response.choices[0].message.content or "").strip()
        if content.startswith("```"):
            content = content.strip("`").strip()
            if content.lower().startswith("json"):
                content = content[4:].strip()
        return json.loads(content)

    def _add_to_allowlist(self, command):
        """将用户手动批准的 WARNING 指令持久化到白名单"""
        allow_file = os.path.join(self.workspace_path, "allowlist", "approved.md")
        try:
            if not os.path.exists(os.path.dirname(allow_file)):
                os.makedirs(os.path.dirname(allow_file))
            with open(allow_file, 'a', encoding='utf-8') as f:
                f.write(f"\n- 已批准指令: {command}")
            logger.info(f"[SYSTEM] 指令已加入持久化白名单")
            self.context_manager.refresh_base()
        except Exception as e:
            logger.error(f"[SYSTEM] 写入白名单失败: {e}")
