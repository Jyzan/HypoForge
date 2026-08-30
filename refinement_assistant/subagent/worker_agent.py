# subagent/worker_agent.py
import html
import json
import time
import threading
from dataclasses import dataclass
from tools.registry import TOOL_REGISTRY
from logger_manager import logger

@dataclass
class WorkerTaskResult:
    summary_for_model: str
    trace_for_ui: str
    final_feedback: str = ""

    def __str__(self):
        return self.summary_for_model


def _clip_text(text, limit=1200):
    text = str(text or "")
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[truncated: full trace is available in execution log]"


class WorkerAgent:
    def __init__(self, client, endpoint_id, token_manager, security_agent, pending_approvals, permission_manager=None):
        self.client = client
        self.endpoint_id = endpoint_id
        self.token_manager = token_manager
        self.security_agent = security_agent
        self.pending_approvals = pending_approvals
        self.permission_manager = permission_manager

    def _wait_for_approval_event(self, event, timeout=300, poll_interval=0.25):
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

    def execute_task(self, task_description, tool_definitions, active_tools, allowlist):
        # 【取消截断 1】：完整打印将军下发给士兵的原始指令
        logger.info(f"\n" + "="*60 + f"\n[WORKER MISSION START] 士兵接收接力任务:\n{task_description}\n" + "="*60)
        yield json.dumps({"type": "status", "content": f"👷 [执行代理接管任务] 开启分段接力模式..."})
        
        system_prompt = (
            "你是一个无统筹规划权的底层执行器(Worker)。将军为你提供了详尽的【SOP执行手册】。\n"
            "【执行与接力纪律】：\n"
            "1. 你的记忆是分段的！每次你都会看到【总指令】和【上一步的交接结果】。\n"
            "2. ⚠️【进度对齐法则（极度重要）】：不要把你当前的“执行回合数(Round)”和将军SOP中的“步骤(Step)”混淆！哪怕你现在是第3个执行回合，如果判断出 SOP 的 Step 1 还没做完，你必须继续做 Step 1！绝对禁止因为回合数的增加而强行跳步！只有上一回合的战果真正达成了当前 SOP 步骤的目标，你才能推进到 SOP 的下一步。\n"
            "3. ⚠️【禁止盲狙法则】：若无确切坐标数据，绝对禁止凭空猜测坐标！必须先调用 analyze_image 寻找！\n"
            "4. ⚠️【情报附言法则】：移交接力棒时，必须将未来可能用到的关键数据（如后续目标的坐标）附加在反馈末尾！\n"
            "5. 🛑【绝对诚实汇报纪律】：严禁脑补、幻觉或提前宣告尚未执行的步骤！没调用的工具就是没做！如果因为报错或回合耗尽导致后续步骤未执行，必须明确声明“后续步骤未执行”！\n"
            "6. 熔断机制：遇到严重报错或无法继续，请直接总结陈词并停止调用工具。\n"
        )

        max_steps = 15  
        step_round = 0  # 变量名在心里变成回合
        
        all_step_reports = []
        progress_tracker = []   
        last_step_feedback = "" 
        tool_usage = []
        
        final_report = "任务未能产生有效结果"

        final_feedback = final_report
        status = "incomplete"
        failed_tool_results = {}

        try:
            while step_round < max_steps:
                step_round += 1
                
                if not tool_definitions:
                    yield json.dumps({"type": "status", "content": "⚠️ 执行代理没有可用工具，任务中止"})
                    return "失败：未挂载任何有效工具。"

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"【将军的原始SOP总指令】:\n{task_description}"}
                ]
                
                if progress_tracker:
                    prog_str = "\n".join(progress_tracker)
                    # 提示词里也改成回合
                    messages.append({"role": "user", "content": f"【系统备忘录：已完成的历史回合】:\n{prog_str}\n(请根据真实战果判断当前应该执行SOP的哪一步)"})
                    
                if last_step_feedback:
                    messages.append({"role": "user", "content": f"【上一回合移交的反馈数据】:\n{last_step_feedback}\n\n请基于此反馈决定本回合操作。"})

                response = self.client.chat.completions.create(
                    model=self.endpoint_id,
                    messages=messages,
                    tools=tool_definitions,
                    tool_choice="auto",
                    temperature=0.1
                )
                self.token_manager.update_usage(response, source=f"WorkerAgent-Round{step_round}")
                msg = response.choices[0].message
                
                if msg.tool_calls:
                    current_step_raw_results = []
                    
                    for tool_call in msg.tool_calls:
                        t_name = tool_call.function.name
                        t_args = tool_call.function.arguments
                        
                        # 日志和界面提示全面替换为“回合”
                        logger.info(f"[WORKER TOOL CALL] 第 {step_round} 回合 | 工具: {t_name} | 参数: {t_args}")
                        yield json.dumps({"type": "status", "content": f"  🛠️ 执行代理(第 {step_round} 回合) 使用: {t_name}"})
                        
                        tool_usage.append(t_name)
                        signature = self._tool_failure_signature(tool_call)
                        if signature and signature in failed_tool_results:
                            tool_result_content = self._duplicate_failure_observation(
                                failed_tool_results[signature]
                            )
                            logger.info("[WORKER TOOL CALL] suppressed duplicate failed tool call")
                        else:
                            tool_result_content = yield from self._run_worker_tool(tool_call, active_tools, allowlist)
                            if signature and self._looks_like_failed_tool_result(tool_result_content):
                                failed_tool_results[signature] = tool_result_content
                        logger.info(f"[WORKER TOOL RESULT] 返回了 {len(tool_result_content)} 个字符。")
                        
                        current_step_raw_results.append(f"[{t_name} 执行结果]:\n{tool_result_content}")
                        progress_tracker.append(f"第 {step_round} 回合: 成功调用了 {t_name}")
                    
                    last_step_feedback = "\n\n".join(current_step_raw_results)
                    all_step_reports.append(f"### 第 {step_round} 回合操作 ###\n{last_step_feedback}")
                    
                    if step_round == max_steps:
                        yield json.dumps({"type": "status", "content": "⚠️ [达到回合上限] 强制汇总..."})
                        final_report = (
                            "【系统强制终止】达到最大执行回合。以下是各回合真实操作汇报汇总：\n" 
                            + "\n".join(all_step_reports) 
                            + "\n\n🚨 【系统强行附注】：执行已因回合耗尽而物理中断！SOP中剩余的指令均未执行！"
                        )
                        final_feedback = final_report
                        status = "partial"
                        break 
                    
                    continue 
                else:
                    final_msg = msg.content
                    all_step_reports.append(f"### 最终结论 ###\n{final_msg}")
                    final_report = "\n".join(all_step_reports)
                    final_feedback = final_msg or final_report
                    status = "completed"
                    yield json.dumps({"type": "status", "content": "🏁 [接力完成] 正在汇总执行报告"})
                    break
                    
        except Exception as e:
            logger.error(f"[WORKER ERROR] 士兵阵亡: {e}", exc_info=True)
            final_report = f"执行过程中发生异常，任务中断。已收集的情报：\n" + "\n".join(all_step_reports) + f"\n\n异常信息: {str(e)}"

        # 【取消截断 2】：完整打印士兵发回给将军的最终长篇战报
        logger.info(f"\n" + "-"*60 + f"\n[WORKER MISSION END] 汇总战报提交:\n{final_report}\n" + "-"*60)
        return self._build_task_result(
            task_description,
            final_report,
            final_feedback,
            progress_tracker,
            tool_usage,
            status=status,
            rounds=step_round,
        )

    def _build_task_result(
        self,
        task_description,
        trace_for_ui,
        final_feedback,
        progress_tracker,
        tool_usage,
        status,
        rounds,
    ):
        unique_tools = []
        for tool_name in tool_usage:
            if tool_name not in unique_tools:
                unique_tools.append(tool_name)
        public_feedback = self._sanitize_user_feedback(final_feedback)
        summary_lines = [
            "[Worker Summary]",
            f"task: {_clip_text(task_description, 500)}",
            f"status: {status}",
            f"rounds: {rounds}",
            f"tools_used: {', '.join(unique_tools) if unique_tools else 'none'}",
            f"progress: {_clip_text('; '.join(progress_tracker), 700) if progress_tracker else 'none'}",
            f"result: {_clip_text(public_feedback, 1200)}",
            "full_trace: retained for UI/logs only; do not request or repeat it unless needed.",
        ]
        return WorkerTaskResult(
            summary_for_model="\n".join(summary_lines),
            trace_for_ui=str(trace_for_ui or ""),
            final_feedback=_clip_text(public_feedback, 1500),
        )

    def _tool_failure_signature(self, tool_call):
        try:
            args = json.loads(tool_call.function.arguments)
        except Exception:
            args = {}
        if tool_call.function.name == "run_powershell" and isinstance(args.get("command"), str):
            args["command"] = html.unescape(args["command"]).strip()
        return (tool_call.function.name, json.dumps(args, sort_keys=True, ensure_ascii=False))

    def _looks_like_failed_tool_result(self, result):
        text = str(result or "").lower()
        failure_markers = [
            "退出代码: 1",
            "exit code: 1",
            "failed",
            "error:",
            "no matching distribution",
            "failed to build",
            "权限受限",
            "执行失败",
        ]
        return any(marker in text for marker in failure_markers)

    def _duplicate_failure_observation(self, previous_result):
        return (
            "[internal-observation: duplicate-failed-tool-call-suppressed]\n"
            "内部观察：重复失败工具调用已被抑制。\n"
            "同一工具调用在当前任务中已经失败过，系统没有再次执行它。\n"
            "上次失败摘要："
            + _clip_text(previous_result, 500)
            + "\n请不要再次执行完全相同的命令；请改用替代方案，或向用户汇报阻塞原因。\n"
            "[/internal-observation]"
        )

    def _sanitize_user_feedback(self, text):
        text = str(text or "")
        start_marker = "[internal-observation:"
        end_marker = "[/internal-observation]"
        while True:
            start = text.find(start_marker)
            if start < 0:
                break
            block_start = text.rfind("\n", 0, start) + 1
            end = text.find(end_marker, start)
            if end < 0:
                line_end = text.find("\n", start)
                block_end = len(text) if line_end < 0 else line_end + 1
            else:
                block_end = end + len(end_marker)
                if block_end < len(text) and text[block_end:block_end + 1] == "\n":
                    block_end += 1
            text = text[:block_start] + text[block_end:]
        return text.strip()

    def _run_worker_tool(self, tool_call, active_tools, allowlist):
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        if name == "run_powershell" and isinstance(args.get("command"), str):
            args["command"] = html.unescape(args["command"])
        MAX_CHAR_LIMIT = 1500  
        
        if name not in active_tools or name not in TOOL_REGISTRY:
            res_content = f"❌ 执行失败：将军并未为你配备 '{name}' 工具。"
            yield json.dumps({"type": "status", "content": res_content})
            return res_content

        permission_gate_approved = False
        if self.permission_manager:
            permission_decision = self.permission_manager.evaluate(name, args)
            if permission_decision.action == "deny":
                res_content = f"权限拒绝: {permission_decision.reason} | {permission_decision.message}"
                yield json.dumps({"type": "status", "content": res_content})
                return res_content
            if permission_decision.action == "ask":
                event = threading.Event()
                self.pending_approvals[tool_call.id] = {"event": event, "approved": False}
                yield json.dumps({
                    "type": "approval",
                    "tool_name": name,
                    "command": permission_decision.command or json.dumps(args, ensure_ascii=False),
                    "tool_call_id": tool_call.id,
                    "audit_type": permission_decision.audit_type,
                    "sandbox_path": permission_decision.sandbox_path,
                    "message": permission_decision.message,
                })
                yield json.dumps({"type": "status", "content": "等待用户确认权限边界..."})
                self._wait_for_approval_event(event, timeout=300)
                approval_data = self.pending_approvals.pop(tool_call.id, None)
                if not approval_data or not approval_data.get("approved"):
                    return f"权限受限: 用户未批准 {permission_decision.reason}"
                permission_gate_approved = True
                if permission_decision.audit_type == "SANDBOX":
                    permission_decision = self.permission_manager.evaluate(name, args)
                    if permission_decision.action == "deny":
                        res_content = f"权限拒绝: {permission_decision.reason} | {permission_decision.message}"
                        yield json.dumps({"type": "status", "content": res_content})
                        return res_content
                    if permission_decision.action == "ask":
                        event = threading.Event()
                        self.pending_approvals[tool_call.id] = {"event": event, "approved": False}
                        yield json.dumps({
                            "type": "approval",
                            "tool_name": name,
                            "command": permission_decision.command or json.dumps(args, ensure_ascii=False),
                            "tool_call_id": tool_call.id,
                            "audit_type": permission_decision.audit_type,
                            "sandbox_path": permission_decision.sandbox_path,
                            "message": permission_decision.message,
                        })
                        yield json.dumps({"type": "status", "content": "等待用户确认危险操作..."})
                        self._wait_for_approval_event(event, timeout=300)
                        approval_data = self.pending_approvals.pop(tool_call.id, None)
                        if not approval_data or not approval_data.get("approved"):
                            return f"权限受限: 用户未批准 {permission_decision.reason}"
                        permission_gate_approved = True
                if permission_decision.rewritten_args:
                    args.update(permission_decision.rewritten_args)

        if name == "run_powershell":
            cmd = args.get("command", "")
            full_trust = bool(getattr(self.permission_manager, "config", {}).get("full_trust_mode"))
            audit_res = (
                "SAFE"
                if (permission_gate_approved or full_trust)
                else self.security_agent.audit(cmd, allowlist)
            )
            
            if audit_res == "DANGER" or audit_res == "WARNING":
                logger.warning(f"[WORKER SECURITY] 士兵根据审计结论拦截: {audit_res} | 指令: {cmd}")
                event = threading.Event()
                self.pending_approvals[tool_call.id] = {"event": event, "approved": False}
                yield json.dumps({
                    "type": "approval", 
                    "tool_name": name,
                    "command": cmd, 
                    "tool_call_id": tool_call.id,
                    "audit_type": audit_res
                })
                
                yield json.dumps({"type": "status", "content": "⏸️ 危险操作，正在原地挂起等待指挥官审批..."})
                
                self._wait_for_approval_event(event, timeout=300) 
                
                approval_data = self.pending_approvals.pop(tool_call.id, None)
                if not approval_data or not approval_data.get("approved"):
                    return f"⚠️ 权限受限：指挥官已拒绝该危险操作。请停止重试。"
                
                yield json.dumps({"type": "status", "content": "✅ 用户已批准，继续执行..."})
            
            res_content = ""
            for line in TOOL_REGISTRY[name]["function"](**args):
                if len(res_content) > MAX_CHAR_LIMIT:
                    res_content += "\n[SYSTEM: 警告] 输出数据量极其巨大，已触发强制截断以保护 Token！"
                    yield json.dumps({"type": "status", "content": "  > ⚠️ 输出过长，触发防爆截断"})
                    break 
                    
                res_content += line + "\n"
                # 【取消截断 3】：让前端也能看到完整的单行执行信息
                yield json.dumps({"type": "status", "content": f"  > 执行: {line}"})
            return res_content
        else:
            func = TOOL_REGISTRY[name]["function"]
            try:
                raw_res = func(**args)
                if isinstance(raw_res, tuple) and len(raw_res) == 2:
                    res_content_str, api_response = raw_res
                    if hasattr(api_response, 'usage'):
                        self.token_manager.update_usage(api_response, source=f"Tool-{name}")
                    res_content = str(res_content_str)
                else:
                    res_content = str(raw_res)
                
                if len(res_content) > MAX_CHAR_LIMIT:
                    res_content = res_content[:MAX_CHAR_LIMIT] + "\n[SYSTEM: 警告] 工具输出过长，已强制截断！"
                yield json.dumps({"type": "status", "content": f"  > 士兵使用 {name} 成功"})
                return res_content
            except Exception as e:
                res_content = f"工具内部异常: {str(e)}"
                yield json.dumps({"type": "status", "content": f"  > 士兵使用 {name} 失败"})
                return res_content
