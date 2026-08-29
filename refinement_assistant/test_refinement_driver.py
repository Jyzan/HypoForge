# test_refinement_driver.py — 以测试用户身份驱动细化对话（字段测试用）
import json
import sys
import urllib.request

BASE = "http://127.0.0.1:5000"

# 只批准这些明显只读的命令模式；其余一律拒绝（生成脚本+说明的兜底路径）
BENIGN_PATTERNS = ["--version", "Get-ChildItem", "Get-Command", "dir ", "Get-Location"]


def post_json(path, payload, timeout=600):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=timeout)


def stream_chat(message):
    """发送一条消息，处理审批请求，返回 final 内容"""
    final_content = ""
    approvals = []
    with post_json("/chat", {"message": message}) as resp:
        buf = b""
        while True:
            chunk = resp.read1(4096) if hasattr(resp, "read1") else resp.read(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n\n" in buf:
                line, buf = buf.split(b"\n\n", 1)
                line = line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                try:
                    data = json.loads(line[6:])
                except Exception:
                    continue
                t = data.get("type")
                if t == "approval":
                    cmd = data.get("command", "")
                    benign = any(p in cmd for p in BENIGN_PATTERNS)
                    print(f"  [审批请求] {cmd[:100]} -> {'批准' if benign else '拒绝'}")
                    approvals.append((cmd, benign))
                    post_json("/approve", {
                        "tool_call_id": data.get("tool_call_id"),
                        "command": cmd,
                        "approved": benign,
                        "audit_type": data.get("audit_type"),
                    }, timeout=30)
                elif t == "final":
                    final_content = data.get("content", "")
                    print(f"  [final] tokens={data.get('tokens')}")
                elif t == "error":
                    print(f"  [error] {data.get('content', '')[:150]}")
    return final_content, approvals


if __name__ == "__main__":
    message = sys.argv[1] if len(sys.argv) > 1 else "继续"
    print(f"== 发送: {message[:80]}...")
    content, approvals = stream_chat(message)
    print("== 回复摘要 ==")
    print(content[:1500])
