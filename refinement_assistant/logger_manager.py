# logger_manager.py
import os
import logging
from datetime import datetime
from dotenv import load_dotenv

# 显式加载一次环境配置
load_dotenv()

class LoggerManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LoggerManager, cls).__new__(cls)
            cls._instance._setup_logger()
        return cls._instance

    def _setup_logger(self):
        # 1. 路径获取与校验
        workspace = os.getenv("WORKSPACE")
        if not workspace:
            # 兜底：如果环境变量读取失败，使用当前文件所在目录
            workspace = os.path.dirname(os.path.abspath(__file__))
            
        log_dir = os.path.join(workspace, "log")
        if not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir)
            except Exception as e:
                print(f"无法创建日志目录: {e}")

        # 2. 获取 Logger 实例
        self.logger = logging.getLogger("AI_Assistant")
        self.logger.setLevel(logging.INFO)
        
        # 清除现有的 handlers，防止重复打印或冲突
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # 3. 设置格式：增强可读性
        log_format = '%(asctime)s [%(levelname)-8s] %(message)s'
        formatter = logging.Formatter(log_format, '%Y-%m-%d %H:%M:%S')

        # 4. 文件处理器 (以日期命名)
        today_str = datetime.now().strftime("%Y-%m-%d")
        log_path = os.path.join(log_dir, f"{today_str}.log")
        
        try:
            fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
        except Exception as e:
            print(f"无法初始化文件日志: {e}")

        # 5. 控制台处理器
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)
        
        # 防止日志向上层传播
        self.logger.propagate = False

        # --- 核心改进 1：屏蔽 Flask(Werkzeug) 的 HTTP 请求日志 ---
        logging.getLogger('werkzeug').setLevel(logging.WARNING)

        # --- 核心改进 2：添加启动分隔符 ---
        start_time = datetime.now().strftime("%H:%M:%S")
        separator = "\n" + "="*80 + f"\n[SYSTEM START] 启动时间: {start_time}\n" + "="*80
        self.logger.info(separator)

    def get_logger(self):
        return self.logger

# 初始化单例
logger = LoggerManager().get_logger()