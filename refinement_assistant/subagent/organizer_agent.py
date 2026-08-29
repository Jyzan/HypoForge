# subagent/organizer_agent.py
import os
import json
import re
from logger_manager import logger

class OrganizerAgent:
    def __init__(self, client, endpoint_id, token_manager, workspace_path):
        self.client = client
        self.endpoint_id = endpoint_id
        self.token_manager = token_manager
        self.skills_dir = os.path.join(workspace_path, "skills")
        self.workspace_path = workspace_path

    def _read_file_safe(self, file_path):
        encodings = ['utf-8', 'utf-8-sig', 'utf-16', 'gbk']
        for enc in encodings:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    return f.read()
            except:
                continue
        return None

    def organize(self, item, details=""):
        """主路由：通过 yield from 将子任务的流式数据透传给 main.py"""
        if item == "skills":
            yield from self._organize_skills(details)
        elif item == "whitebook":
            yield from self._organize_whitebook(details)
        else:
            yield json.dumps({"type": "error", "content": f"未知的整理项目: {item}"})

    def _organize_whitebook(self, details):
        whitebook_path = os.path.join(self.workspace_path, "allowlist", "approved.md")
        if not os.path.exists(whitebook_path):
            yield json.dumps({"type": "error", "content": "白名单文件不存在。"})
            return

        content = self._read_file_safe(whitebook_path)
        
        prompt = (
            "你是一个安全审计专家。请整理以下【白名单指令集】：\n"
            "【任务要求】：\n"
            "1. **去重**：删除语义完全重复的指令。\n"
            "2. **清洗**：剔除那些包含大段源代码、冗长的配置文件内容的指令（这些通常是误记录）。只保留简短、通用的操作指令。\n"
            "3. **规范化**：确保每条指令简洁明了。\n"
            "4. **额外指令**：" + (details if details else "无") + "\n\n"
            "【原始白名单内容】：\n" + content + "\n\n"
            "请直接返回整理后的完整文件内容，不要包含任何解释文字或 Markdown 代码块标签。"
        )

        try:
            yield json.dumps({"type": "status", "content": "正在清洗白名单规则..."})
            
            resp = self.client.chat.completions.create(
                model=self.endpoint_id,
                messages=[{"role": "system", "content": prompt}],
                temperature=0.2
            )
            self.token_manager.update_usage(resp, source="Organizer-Whitebook")
            new_content = resp.choices[0].message.content.strip()

            if new_content:
                with open(whitebook_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                logger.info("[ORGANIZER] 白名单整理完成。")
                
                yield json.dumps({
                    "type": "final", 
                    "content": "📜 **白名单 (Whitebook) 整理完成！**\n- 已删除重复条目\n- 已剔除冗余代码块指令\n- 权限表已精简。",
                    "tokens": self.token_manager.get_status_report()
                })
            else:
                yield json.dumps({"type": "error", "content": "模型返回的内容为空。"})
                
        except Exception as e:
            logger.error(f"[ORGANIZER] 白名单整理失败: {e}")
            yield json.dumps({"type": "error", "content": f"整理失败: {str(e)}"})

    def _organize_skills(self, details):
        if not os.path.exists(self.skills_dir):
            yield json.dumps({"type": "error", "content": "skills 目录不存在，无需整理。"})
            return

        files_content = {}
        # 1. 读取文件（严格排除 base.md 和旧的 catalogue.md）
        for filename in os.listdir(self.skills_dir):
            if filename.endswith(".md") and filename not in ["base.md", "catalogue.md"]:
                file_path = os.path.join(self.skills_dir, filename)
                content = self._read_file_safe(file_path)
                if content is not None:
                    files_content[filename] = content
                else:
                    logger.warning(f"[ORGANIZER] 无法解析文件编码，跳过: {filename}")

        if not files_content:
            yield json.dumps({"type": "final", "content": "skills 目录下没有需要整理的技能文件。"})
            return

        # 2. 构造子代理的 Prompt
        prompt = f"""你是一个专门负责整理和压缩AI记忆（Markdown文件）的子代理。
当前任务：对以下提供的技能文件内容进行分类、去重、整合，并生成新的分类好的Markdown文件。
用户的额外要求（如果有，必须严格遵守）：{details if details else '无'}

你需要严格遵循以下原则：
1. 根据内容的逻辑关联性，将它们归类到 1-4 个核心维度（例如：environment_setup.md, code_fixes.md 等）。
2. 删除重复的、废话的、错误的经验，只保留精炼的干货和正确可执行的步骤。
3. "files_to_delete" 列表里只包含你已经成功吸纳了内容、可以被安全删除的旧文件名。
4. 尽量用英文起名新文件，保持简洁明了，统一后缀为 .md。
5. 【核心任务】：你必须在 new_files 列表中生成一个名为 catalogue.md 的文件。这是所有技能文件的总目录检索！你必须在里面列出 base.md 以及你本次生成的所有新分类文件，并用一句话简述每个文件里记载了什么功能经验。主模型将依赖这个目录来判断需要读取哪个技能。
6. 绝对禁止将 base.md 和 catalogue.md 放入删除列表。如果你决定更新某个文件且保持原名不变，绝对不要把它放进 files_to_delete！

【当前存在的技能文件与内容】：
{json.dumps(files_content, ensure_ascii=False, indent=2)}

必须且只能输出合法的 JSON 格式，不要包含任何额外的解释说明，格式如下：
{{
    "new_files": [
        {{"filename": "catalogue.md", "content": "# 技能目录\\n- base.md: 核心准则\\n- 新分类1.md: 简述..."}},
        {{"filename": "新分类名1.md", "content": "# 分类1标题\\n...整合后的精简内容..."}}
    ],
    "files_to_delete": ["旧文件1.md", "旧文件2.md"]
}}
"""
        try:
            # 增加一个中间状态的流式输出，安抚用户的等待焦虑
            yield json.dumps({"type": "status", "content": "正在阅读并重组技能文档，这可能需要几十秒..."})
            
            response = self.client.chat.completions.create(
                model=self.endpoint_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1 
            )
            self.token_manager.update_usage(response, source="Organizer-Skills")
            result_text = response.choices[0].message.content.strip()

            # 3. 解析清洗 JSON
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if not json_match:
                raise ValueError("模型未返回有效的 JSON 结构")
            
            result_json = json.loads(json_match.group())

            # 4. 物理文件操作
            created_files = []
            deleted_files = []

            # 先写入新文件
            for new_file in result_json.get("new_files", []):
                new_filename = new_file.get("filename")
                if new_filename == "base.md": continue 
                
                file_path = os.path.join(self.skills_dir, new_filename)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_file.get("content", ""))
                created_files.append(new_filename)

            # 后删除旧文件
            for old_filename in result_json.get("files_to_delete", []):
                # 严格保护核心文件与刚刚生成的新文件
                if old_filename in ["base.md", "catalogue.md"]: continue 
                if old_filename in created_files: continue
                
                file_path = os.path.join(self.skills_dir, old_filename)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    deleted_files.append(old_filename)

            logger.info(f"[ORGANIZER] 整理完成。新增: {created_files}, 删除: {deleted_files}")
            
            msg_lines = ["**记忆模块整理完成！目录已更新。**"]
            if created_files: msg_lines.append(f"✨ **新增/更新文件**：{', '.join(created_files)}")
            if deleted_files: msg_lines.append(f"🗑️ **移除冗余文件**：{', '.join(deleted_files)}")
            
            # 将最终结果抛给前端
            yield json.dumps({
                "type": "final", 
                "content": "\n".join(msg_lines),
                "tokens": self.token_manager.get_status_report()
            })

        except Exception as e:
            logger.error(f"[ORGANIZER SUB-AGENT] 整理异常: {e}")
            yield json.dumps({"type": "error", "content": f"整理过程中出现异常: {str(e)}"})