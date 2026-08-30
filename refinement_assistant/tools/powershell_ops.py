# tools/powershell_ops.py
import os
import subprocess
import threading

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


# 单条命令整体超时（秒）。超时后强杀整个进程树（含 uv/python 等孙进程）
COMMAND_TIMEOUT_SECONDS = int(os.getenv("POWERSHELL_TIMEOUT_SECONDS", "900"))


def _kill_tree(pid):
    """Windows 下按进程树强杀（/T），避免 uv/python 等子进程存活占住输出管道"""
    try:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
        )
    except Exception:
        pass


def run_powershell(command: str):
    """
    执行 PowerShell 命令并以生成器形式实时返回输出。
    【安全升级】：自带防爆盾，最大输出 150 行；整体超时强杀进程树；
    强制 UTF-8 输入输出，避免中文内容被 GBK 编码破坏。
    """
    max_lines = 150  # 严格限制最大输出行数
    
    try:
        if _is_cancelled():
            yield "[SYSTEM: cancel] PowerShell command skipped because task was cancelled."
            return
        # 开启进程
        # 强制 UTF-8：PowerShell 5.1 默认按系统 GBK 编码输出/写文件，中文内容会被破坏
        full_command = (
            "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; "
            "$OutputEncoding=[System.Text.Encoding]::UTF8; " + command
        )
        process = subprocess.Popen(
            ["powershell", "-Command", full_command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # 将错误也合并到输出流
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1 # 行缓冲
        )

        # 看门狗：整体超时后强杀整个进程树，防止 uv/python 等子进程占住管道导致永久阻塞
        watchdog = threading.Timer(
            COMMAND_TIMEOUT_SECONDS, _kill_tree, args=(process.pid,)
        )
        watchdog.daemon = True
        watchdog.start()

        line_count = 0
        # 实时读取每一行输出
        for line in iter(process.stdout.readline, ''):
            if _is_cancelled():
                _kill_tree(process.pid)
                yield "[SYSTEM: cancel] 当前任务已终止，PowerShell 进程树已结束。"
                break
            if line:
                yield line.strip()
                line_count += 1

                # 【防爆核心】达到行数上限，立刻终止子进程
                if line_count >= max_lines:
                    yield f"\n[SYSTEM: 警告] 输出已超过 {max_lines} 行，为防止 Token 爆炸与进程卡死，已强制截断并终止命令！"
                    _kill_tree(process.pid)
                    break

        watchdog.cancel()
        process.stdout.close()
        # 等待进程完全退出（加上超时防止僵尸进程）
        try:
            return_code = process.wait(timeout=5)
            # 如果是被我们主动 kill 的，return_code 可能是负数，不报异常退出码
            if return_code != 0 and line_count < max_lines:
                yield f"--- 命令执行结束，退出代码: {return_code} ---"
        except subprocess.TimeoutExpired:
            _kill_tree(process.pid)
            yield "--- [SYSTEM: 进程超时，已强制清理] ---"
            
    except Exception as e:
        yield f"系统异常: {str(e)}"
