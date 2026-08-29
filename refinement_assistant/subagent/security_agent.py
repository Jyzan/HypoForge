# subagent/security_agent.py
import os
import json
import hashlib
from logger_manager import logger

class SecurityAgent:
    def __init__(self, client, endpoint_id, token_manager):
        self.client = client
        self.endpoint_id = endpoint_id
        self.token_manager = token_manager
        
        # 初始化本地存储路径
        workspace = os.getenv("WORKSPACE", ".")
        self.cache_file = os.path.join(workspace, "allowlist", "audit_cache.json")
        self.greenlist_file = os.path.join(workspace, "allowlist", "greenlist.json")
        
        # 加载数据（绿名单使用 set 集合，查询速度 O(1)，比 for 循环更快）
        self.cache = self._load_json(self.cache_file, {})
        self.green_list = set(self._load_json(self.greenlist_file, []))

    def _load_json(self, filepath, default_val):
        """通用的 JSON 读取工具"""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"读取文件 {filepath} 失败: {e}")
        return default_val

    def _save_json(self, data, filepath):
        """通用的 JSON 保存工具"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            # 如果是 set 集合，保存前需转换为 list
            save_data = list(data) if isinstance(data, set) else data
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"保存文件 {filepath} 失败: {e}")

    def audit(self, command, allowlist_content):
        cmd_stripped = command.strip()

        # =========================================================
        # 第一道防线：物理白名单 (你亲自同意的)
        # =========================================================
        if cmd_stripped in allowlist_content:
            logger.info(f"[SECURITY DIRECT] 物理白名单命中，直接放行: {command}")
            return "SAFE"

        # =========================================================
        # 第二道防线：物理绿名单 (AI 曾判定为 SAFE 的)
        # =========================================================
        if cmd_stripped in self.green_list:
            logger.info(f"[SECURITY GREEN] 绿名单命中，直接放行: {command}")
            return "SAFE"

        # =========================================================
        # 第三道防线：MD5 AI 缓存 (专门拦截重复的 WARNING / DANGER)
        # =========================================================
        raw_str = f"{command}||{allowlist_content}"
        fingerprint = hashlib.md5(raw_str.encode('utf-8')).hexdigest()
        
        if fingerprint in self.cache:
            decision = self.cache[fingerprint]
            logger.info(f"[SECURITY CACHE HIT] 命中智能缓存: {decision} | 指令: {command}")
            return decision

        # =========================================================
        # 第四道防线：花 Token 呼叫 AI 审计
        # =========================================================
        audit_prompt = (
            f"作为安全员，根据以下【白名单】判定指令风险：\n"
            f"【白名单内容】\n{allowlist_content}\n\n"
            f"指令：{command}\n"
            f"判定规则：\n"
            f"1. 若指令在白名单中或绝对安全（如单纯的文件查看），回答 'SAFE'\n"
            f"2. 若指令具有破坏性（删系统文件、修改代码等），回答 'DANGER'\n"
            f"3. 若指令属低风险但有未知副作用（如执行脚本、安装库），回答 'WARNING'\n\n"
            f"【最高指令】：你只能输出 'SAFE', 'WARNING', 'DANGER' 这三个词中的一个！绝不允许输出任何标点符号、换行或解释性文字！"
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.endpoint_id,
                messages=[{"role": "user", "content": audit_prompt}],
                temperature=0.0, 
                max_tokens=5  
            )
            self.token_manager.update_usage(response, source="Security-Audit")
            
            decision_raw = response.choices[0].message.content.strip().upper()
            
            if "DANGER" in decision_raw or "有" in decision_raw: 
                decision = "DANGER"
            elif "WARNING" in decision_raw or "警告" in decision_raw: 
                decision = "WARNING"
            else: 
                decision = "SAFE"
                
            logger.info(f"[SECURITY SUB-AGENT] AI首次判定: {decision} | 指令: {command}")
            
            # =========================================================
            # 分流存储：安全进绿名单，危险进 MD5 缓存
            # =========================================================
            if decision == "SAFE":
                self.green_list.add(cmd_stripped)
                self._save_json(self.green_list, self.greenlist_file)
            else:
                self.cache[fingerprint] = decision
                self._save_json(self.cache, self.cache_file)
            
            return decision
        except Exception as e:
            logger.error(f"[SECURITY SUB-AGENT] 审计异常: {e}")
            return "DANGER"