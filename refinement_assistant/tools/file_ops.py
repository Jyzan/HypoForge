# tools/file_ops.py
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# 读取沙箱配置
SANDBOX_SPACE = os.getenv("SANDBOX_SPACE") or os.path.join(os.path.dirname(__file__), "..", "sandbox")
SANDBOX_ENABLE = os.getenv("SANDBOX_ENABLE", "True").lower() == "true"

# 确保沙箱目录存在
if SANDBOX_SPACE and not os.path.exists(SANDBOX_SPACE):
    os.makedirs(SANDBOX_SPACE)

def _get_safe_path(filename):
    """安全路径转换：将相对路径转换为沙箱内的绝对路径，并防止路径穿越攻击"""
    if not SANDBOX_ENABLE:
        return filename
    
    # 强制将文件限制在沙箱目录下
    # 移除路径前面的斜杠，确保它被视为相对路径
    clean_filename = filename.lstrip(r'\/')
    safe_path = os.path.abspath(os.path.join(SANDBOX_SPACE, clean_filename))
    
    # 检查最终路径是否仍然在沙箱内
    if not safe_path.startswith(os.path.abspath(SANDBOX_SPACE)):
        raise PermissionError(f"禁止访问沙箱外的路径: {filename}")
    
    return safe_path

def list_files(directory="."):
    """列出目录下的所有文件"""
    try:
        target_path = _get_safe_path(directory)
        files = os.listdir(target_path)
        return f"目录 {directory} 下的文件有: {', '.join(files)}"
    except Exception as e:
        return f"列出文件失败: {str(e)}"

def create_file(filename, content=""):
    """创建新文件并写入内容"""
    try:
        target_path = _get_safe_path(filename)
        # 子目录不存在时自动创建（如 scripts/model.py）
        parent_dir = os.path.dirname(target_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"文件创建成功: {target_path}")
        return f"文件 '{filename}' 已成功创建。"
    except Exception as e:
        return f"创建文件失败: {str(e)}"

def read_file(filename):
    """读取文件内容"""
    try:
        target_path = _get_safe_path(filename)
        with open(target_path, "r", encoding="utf-8") as f:
            content = f.read()
        return f"文件 '{filename}' 的内容如下:\n---\n{content}\n---"
    except Exception as e:
        return f"读取文件失败: {str(e)}"

def modify_file(filename, content, mode="append"):
    """修改文件内容：append（追加）或 overwrite（覆盖）"""
    try:
        target_path = _get_safe_path(filename)
        write_mode = "a" if mode == "append" else "w"
        with open(target_path, write_mode, encoding="utf-8") as f:
            f.write(content if mode == "overwrite" else "\n" + content)
        return f"文件 '{filename}' 已成功{ '追加' if mode == 'append' else '覆盖' }内容。"
    except Exception as e:
        return f"修改文件失败: {str(e)}"

# 在 file_ops.py 中添加这个函数
def delete_file(filename):
    """删除沙箱中的指定文件"""
    try:
        target_path = _get_safe_path(filename)
        if os.path.exists(target_path):
            os.remove(target_path)
            logger.info(f"文件删除成功: {target_path}")
            return f"文件 '{filename}' 已从沙箱中成功删除。"
        else:
            return f"删除失败：文件 '{filename}' 不存在。"
    except Exception as e:
        return f"删除文件过程中出现错误: {str(e)}"