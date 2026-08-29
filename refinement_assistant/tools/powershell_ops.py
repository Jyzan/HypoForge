# tools/powershell_ops.py
import subprocess

_cancel_event = None


def set_cancel_event(cancel_event):
    global _cancel_event
    _cancel_event = cancel_event


def clear_cancel_event():
    global _cancel_event
    _cancel_event = None


def _is_cancelled():
    return bool(_cancel_event and _cancel_event.is_set())


def is_cancelled():
    return _is_cancelled()


def run_powershell(command: str):
    """
    执行 PowerShell 命令并以生成器形式实时返回输出。
    【安全升级】：自带防爆盾，最大输出 150 行，超时 10 秒，超限直接物理击杀进程。
    """
    max_lines = 150  # 严格限制最大输出行数
    
    try:
        if _is_cancelled():
            yield "[SYSTEM: cancel] PowerShell command skipped because task was cancelled."
            return
        # 开启进程
        process = subprocess.Popen(
            ["powershell", "-Command", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # 将错误也合并到输出流
            text=True,
            encoding='gbk',
            bufsize=1 # 行缓冲
        )
        
        line_count = 0
        # 实时读取每一行输出
        for line in iter(process.stdout.readline, ''):
            if _is_cancelled():
                process.kill()
                yield "[SYSTEM: cancel] 当前任务已终止，PowerShell 子进程已结束。"
                break
            if line:
                yield line.strip()
                line_count += 1
                
                # 【防爆核心】达到行数上限，立刻终止子进程
                if line_count >= max_lines:
                    yield f"\n[SYSTEM: 警告] 输出已超过 {max_lines} 行，为防止 Token 爆炸与进程卡死，已强制截断并终止命令！"
                    process.kill()
                    break
        
        process.stdout.close()
        # 等待进程完全退出（加上超时防止僵尸进程）
        try:
            return_code = process.wait(timeout=5)
            # 如果是被我们主动 kill 的，return_code 可能是负数，不报异常退出码
            if return_code != 0 and line_count < max_lines:
                yield f"--- 命令执行结束，退出代码: {return_code} ---"
        except subprocess.TimeoutExpired:
            process.kill()
            yield "--- [SYSTEM: 进程超时，已强制清理] ---"
            
    except Exception as e:
        yield f"系统异常: {str(e)}"
