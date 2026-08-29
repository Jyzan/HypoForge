# subagent/summarizer_agent.py
import json
import re
from logger_manager import logger

class SummarizerAgent:
    def __init__(self, client, endpoint_id, token_manager):
        self.client = client
        self.endpoint_id = endpoint_id
        self.token_manager = token_manager

    def _extract_json(self, text):
        """通用的 JSON 提取工具函数"""
        try:
            # 优先尝试寻找最外层的 {}
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group())
            return None
        except:
            return None

    def summarize(self, messages):
        """生成深度摘要 (移除不支持的 json_object 参数)"""
        prompt = (
            "你是一个记忆整理专家。请为以下对话生成压缩摘要。\n"
            "注意：必须保留关键信息（如：涉及的文件名、路径、工具名称、具体的代码逻辑）。\n"
            "【特别注意】：如果对话中包含 [士兵战报] 或 [delegate_task] 的返回内容，请务必将战报中的核心数据和路径提取到摘要中！\n"
            "请直接返回一个 JSON 格式的对象，不要输出任何解释文字。\n"
            "格式示例：{\"topic\": \"主题\", \"summary\": \"摘要内容\"}\n\n"
            f"对话内容：\n{json.dumps(messages, ensure_ascii=False)}"
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.endpoint_id,
                messages=[{"role": "system", "content": prompt}],
                # ❌ 删除了 response_format={"type": "json_object"}
                temperature=0.3
            )
            self.token_manager.update_usage(resp, source="Summarizer-Summarize")
            content = resp.choices[0].message.content
            
            data = self._extract_json(content)
            if data:
                return data
            
            logger.warning(f"[SUMMARIZER] 无法从内容中解析JSON，原始输出: {content[:100]}")
            return {"topic": "常规对话", "summary": "摘要解析失败"}
        except Exception as e:
            logger.error(f"[SUMMARIZER] 总结接口调用失败: {e}")
            return {"topic": "未知任务", "summary": "接口异常，无摘要"}

    def rank_relevance(self, blocks, current_query):
        """计算 R 值分布 (移除不支持的 json_object 参数)"""
        if not blocks: return []
        
        summary_list = [f"ID {i}: [主题:{b['topic']}] {b['summary']}" for i, b in enumerate(blocks)]
        
        prompt = (
            f"判定历史任务与当前指令的相关性 R (0.0-1.0)。\n"
            f"当前指令: \"{current_query}\"\n"
            f"历史摘要列表:\n" + "\n".join(summary_list) + "\n\n"
            "只要涉及相同的话题、工具、路径或逻辑，就给高分。请直接返回一个 JSON 数组，如 [0.1, 0.9]"
        )
        
        try:
            resp = self.client.chat.completions.create(
                model=self.endpoint_id,
                messages=[{"role": "system", "content": prompt}],
                temperature=0.1
            )
            self.token_manager.update_usage(resp, source="Summarizer-Rank")
            content = resp.choices[0].message.content
            
            # 寻找数组 [ ... ]
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                r_list = json.loads(match.group())
                if len(r_list) == len(blocks):
                    return r_list
            
            logger.warning(f"[SUMMARIZER] R值解析失败，原始输出: {content[:50]}")
            return [0.5] * len(blocks)
        except Exception as e:
            logger.error(f"[SUMMARIZER] R值判定异常: {e}")
            return [0.0] * len(blocks)