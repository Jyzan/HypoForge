# app.py
from flask import Flask, render_template, request, Response, stream_with_context, jsonify, send_file
from main import AIAssistant
from tools.registry import TOOL_REGISTRY
from core.llm_client import get_chat_model
import os, signal, threading, json, re
from datetime import datetime

app = Flask(__name__)
ai_brain = AIAssistant()

@app.after_request
def _no_store_dynamic(response):
    """动态接口禁止浏览器缓存，防止会话列表/细化产出读到旧数据"""
    if request.path.startswith('/api/') or request.path == '/init':
        response.headers['Cache-Control'] = 'no-store'
    return response

def _archives_dir():
    path = os.path.join(ai_brain.workspace_path, "archives")
    os.makedirs(path, exist_ok=True)
    return path

def _safe_archive_name(name):
    name = re.sub(r"[^\w\-.]+", "_", name, flags=re.UNICODE).strip("._")
    return name or "untitled"

def _read_archive_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.route('/')
def index(): 
    return render_template('index.html')

@app.route('/init')
def init():
    return jsonify({
        "tokens": ai_brain.token_manager.get_status_report(),
        "tokens_used": ai_brain.token_manager.total_used,
        "tokens_limit": ai_brain.token_manager.max_threshold,
        "model": get_chat_model(),
        "refinement_pending": _refinement_kickoff()[0],
    })

# ================= 固定方案自动载入（HypoForge → 细化工作台） =================
def _refinement_input_path():
    return os.path.join(ai_brain.workspace_path, "refinement_input.json")

def _refinement_kickoff():
    """返回 (pending, kickoff 消息, run_id, switch)。
    pending = 存在固定方案且其 run_id 尚未载入过；
    switch = 当前对话非空，载入前需要先保存并重置。"""
    path = _refinement_input_path()
    if not os.path.exists(path):
        return False, None, "", False
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False, None, "", False
    run_id = str(data.get("run_id") or "")
    if run_id == getattr(ai_brain, "current_refinement_run_id", ""):
        return False, None, run_id, False
    switch = bool(ai_brain.context_manager.messages)
    question = str(data.get("question") or "").strip()[:300]
    message = (
        f"[系统] 已自动载入固定研究方案（run_id: {run_id}）。\n"
        f"原始问题：{question}\n\n"
        "请开始对话式细化，按以下流程执行：\n"
        "1. 先用 load_refinement_input 读取完整方案，用几句话总结核心思路；\n"
        "2. 向我确认细化所需的关键信息，例如：我可用的设备与软件环境、算力、"
        "偏好的技术方向、时间预算、是否已安装 FoldX 或相关数据集；\n"
        "3. 结合我的回答生成完整细化计划：材料与试剂清单、实验步骤拆解、时间线、"
        "风险与替代方案、干实验脚本；\n"
        "4. 执行任何命令前必须先向我申请权限，没有权限时只生成脚本和运行说明；\n"
        f"5. 最终用 save_refinement_output 保存结构化 JSON 和 Markdown（run_id 用 {run_id}）。"
    )
    return True, message, run_id, switch

@app.route('/api/session/current')
def current_session_messages():
    """返回当前内存中对话的消息，用于页面刷新后恢复显示"""
    msgs = [
        {"role": m.get("role"), "content": m.get("content", "")}
        for m in ai_brain.context_manager.messages
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]
    sid = getattr(ai_brain, "current_session_id", None)
    title = ""
    if sid:
        try:
            path = os.path.join(_sessions_dir(), _safe_session_file(sid))
            if os.path.exists(path):
                title = _read_archive_file(path).get("title", "")
        except Exception:
            pass
    return jsonify({
        "messages": msgs,
        "tokens": ai_brain.token_manager.get_status_report(),
        "tokens_used": ai_brain.token_manager.total_used,
        "tokens_limit": ai_brain.token_manager.max_threshold,
        "session_id": sid,
        "title": title,
        "refinement_run_id": getattr(ai_brain, "current_refinement_run_id", "") or "",
    })

@app.route('/api/refinement/kickoff')
def refinement_kickoff():
    pending, message, run_id, switch = _refinement_kickoff()
    return jsonify({"pending": pending, "message": message, "run_id": run_id, "switch": switch})

@app.route('/api/refinement/start', methods=['POST'])
def refinement_start():
    """标记该 run_id 的固定方案已开始细化，防止页面刷新后重复 kickoff"""
    run_id = str((request.json or {}).get("run_id", ""))
    ai_brain.current_refinement_run_id = run_id
    # 持久化，重启服务后也不会对同一方案重复 kickoff
    ai_brain.config_manager.config["last_refinement_run_id"] = run_id
    ai_brain.config_manager.save()
    return jsonify({"status": "success"})

# ================= 细化结果展示 =================
def _latest_refinement_output():
    """返回当前对话对应 run 的细化产出；没有则回退到全局最新的一份"""
    root = os.path.abspath(os.path.join(ai_brain.workspace_path, "..", "output", "refined"))

    def _entry(run_id):
        md_path = os.path.join(root, run_id, "refinement.md")
        if os.path.isfile(md_path):
            return {
                "run_id": run_id,
                "md_path": md_path,
                "json_path": os.path.join(root, run_id, "refinement.json"),
                "mtime": os.path.getmtime(md_path),
            }
        return None

    # 优先：当前对话所属的固定方案 run
    preferred = str(getattr(ai_brain, "current_refinement_run_id", "") or "")
    if preferred:
        entry = _entry(preferred)
        if entry:
            return entry

    # 回退：全局最新的一份
    best = None
    if os.path.isdir(root):
        for run_dir in os.listdir(root):
            candidate = _entry(run_dir)
            if candidate and (best is None or candidate["mtime"] > best["mtime"]):
                best = candidate
    return best

@app.route('/api/refinement/output')
def refinement_output():
    best = _latest_refinement_output()
    if best is None:
        return jsonify({"exists": False})
    try:
        with open(best["md_path"], encoding="utf-8", errors="ignore") as f:
            markdown = f.read()
    except OSError as e:
        return jsonify({"exists": False, "error": str(e)})
    json_text = ""
    if os.path.isfile(best["json_path"]):
        try:
            with open(best["json_path"], encoding="utf-8", errors="ignore") as f:
                json_text = f.read()
        except OSError:
            pass
    return jsonify({
        "exists": True,
        "run_id": best["run_id"],
        "markdown": markdown,
        "json_text": json_text,
        "updated_at": datetime.fromtimestamp(best["mtime"]).strftime("%Y-%m-%d %H:%M:%S"),
        "md_path": best["md_path"],
    })

@app.route('/api/refinement/output/download')
def refinement_output_download():
    best = _latest_refinement_output()
    if best is None:
        return jsonify({"error": "还没有细化方案产出"}), 404
    return send_file(
        best["md_path"],
        as_attachment=True,
        download_name=f"refinement_{best['run_id']}.md",
    )

@app.route('/reset', methods=['POST'])
def reset():
    # 重置前先把当前对话存入历史
    _save_current_session()
    ai_brain.reset_chat()
    new_status = ai_brain.token_manager.get_status_report()
    return jsonify({"status": "success", "tokens": new_status})

@app.route('/double_token', methods=['POST'])
def double_token():
    ai_brain.boost_token_limit()
    status = ai_brain.token_manager.get_status_report()
    return jsonify({"status": "success", "tokens": status})

# ================= 核心修复：补回动态设置 Token 的接口 =================
@app.route('/set_token', methods=['POST'])
def set_token():
    data = request.json
    new_limit = data.get('limit')
    if new_limit is not None:
        ai_brain.set_token_limit(new_limit)
    status = ai_brain.token_manager.get_status_report()
    return jsonify({"status": "success", "tokens": status})

@app.route('/api/organize', methods=['POST'])
def organize():
    data = request.json
    item = data.get('item')
    details = data.get('details')
    
    # 核心修复：把生成器产出的每一条 JSON，包装成前端能识别的 SSE 格式
    return Response(
        stream_with_context((f"data: {c}\n\n" for c in ai_brain.organize_yield(item, details))),
        mimetype='text/event-stream'
    )

@app.route('/api/memory', methods=['POST'])
def memory_cmd():
    data = request.json
    mode = data.get('mode')
    name = data.get('name')
    
    # 将生成器产生的流传递给前端
    return Response(
        stream_with_context((f"data: {c}\n\n" for c in ai_brain.memory_yield(mode, name))),
        mimetype='text/event-stream'
    )

@app.route('/api/memories', methods=['GET'])
def get_memories():
    import os
    memories_dir = os.path.join(ai_brain.workspace_path, "memories")
    memories = []
    if os.path.exists(memories_dir):
        # 扫描目录，去掉 .md 后缀，将纯文件名发给前端
        memories = [f[:-3] for f in os.listdir(memories_dir) if f.endswith(".md")]
    return jsonify({"memories": memories})

@app.route('/api/archives', methods=['GET'])
def list_archives():
    archive_items = []
    for filename in os.listdir(_archives_dir()):
        if not filename.endswith(".json"):
            continue
        file_path = os.path.join(_archives_dir(), filename)
        try:
            data = _read_archive_file(file_path)
            archive_items.append({
                "id": filename[:-5],
                "title": data.get("title", filename[:-5]),
                "created_at": data.get("created_at"),
                "message_count": len(data.get("messages", []))
            })
        except Exception:
            continue

    archive_items.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    return jsonify({"archives": archive_items})

@app.route('/api/archives', methods=['POST'])
def save_archive():
    data = request.json or {}
    messages = data.get("messages") or []
    if not messages:
        return jsonify({"status": "error", "message": "No messages to archive"}), 400

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    title = (data.get("title") or f"Archive {stamp}").strip()
    archive_id = _safe_archive_name(f"{stamp}_{title}")[:120]
    payload = {
        "id": archive_id,
        "title": title,
        "created_at": now,
        "messages": messages
    }

    file_path = os.path.join(_archives_dir(), f"{archive_id}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return jsonify({"status": "success", "archive": payload})

@app.route('/api/archives/<archive_id>', methods=['GET'])
def get_archive(archive_id):
    safe_id = _safe_archive_name(archive_id)
    file_path = os.path.join(_archives_dir(), f"{safe_id}.json")
    if not os.path.exists(file_path):
        return jsonify({"status": "error", "message": "Archive not found"}), 404
    return jsonify(_read_archive_file(file_path))


@app.route('/api/archives/<archive_id>/resume', methods=['POST'])
def resume_archive(archive_id):
    safe_id = _safe_archive_name(archive_id)
    file_path = os.path.join(_archives_dir(), f"{safe_id}.json")
    if not os.path.exists(file_path):
        return jsonify({"status": "error", "message": "Archive not found"}), 404

    archive = _read_archive_file(file_path)
    ai_brain.resume_archive(archive.get("messages", []))
    return jsonify({
        "status": "success",
        "archive": archive,
        "tokens": ai_brain.token_manager.get_status_report()
    })

# ================= 历史对话会话管理 =================
def _sessions_dir():
    path = os.path.join(ai_brain.workspace_path, "sessions")
    os.makedirs(path, exist_ok=True)
    return path

def _safe_session_file(session_id):
    """会话 ID 白名单化，防止路径穿越"""
    name = re.sub(r"[^\w\-]+", "_", str(session_id)).strip("._")
    return (name or "untitled") + ".json"

def _save_current_session(touch=True):
    """把当前对话持久化到 sessions/<id>.json，无有效消息时跳过。
    touch=False 用于切换会话时的现场保存：内容更新但不刷新 updated_at，
    避免刚切走的会话在列表里跳到最顶部。"""
    msgs = [m for m in ai_brain.context_manager.messages
            if m.get("role") in ("user", "assistant") and m.get("content")]
    if not msgs:
        return None
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sid = getattr(ai_brain, "current_session_id", None)
    created_at = now
    updated_at = now
    title_custom = False
    existing_title = ""
    if sid:
        path = os.path.join(_sessions_dir(), _safe_session_file(sid))
        if os.path.exists(path):
            try:
                old = _read_archive_file(path)
                created_at = old.get("created_at", now)
                if not touch:
                    updated_at = old.get("updated_at", now)
                title_custom = bool(old.get("title_custom"))
                existing_title = old.get("title", "")
            except Exception:
                pass
    else:
        sid = "sess-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + str(os.getpid())[-4:]
        ai_brain.current_session_id = sid
    if title_custom and existing_title:
        title = existing_title  # 用户改过名，自动保存不覆盖
    else:
        title = next((m["content"][:40] for m in msgs if m["role"] == "user"), now)
    data = {
        "id": sid,
        "title": title,
        "title_custom": title_custom,
        "created_at": created_at,
        "updated_at": updated_at,
        "messages": [{"role": m["role"], "content": m["content"]} for m in msgs],
        "history_blocks": ai_brain.context_manager.history_blocks,
        "tokens": ai_brain.token_manager.get_status_report(),
        "tokens_used": ai_brain.token_manager.total_used,
    }
    path = os.path.join(_sessions_dir(), _safe_session_file(sid))
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f">>> 会话保存失败: {e}")
        return None
    return data

@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    sessions = []
    for fname in os.listdir(_sessions_dir()):
        if not fname.endswith(".json"):
            continue
        try:
            data = _read_archive_file(os.path.join(_sessions_dir(), fname))
        except Exception:
            continue
        sessions.append({
            "id": data.get("id", fname[:-5]),
            "title": data.get("title", "未命名对话"),
            "updated_at": data.get("updated_at", ""),
            "message_count": len(data.get("messages", [])),
        })
    sessions.sort(key=lambda x: x["updated_at"], reverse=True)
    return jsonify({"sessions": sessions})

@app.route('/api/sessions/<session_id>/load', methods=['POST'])
def load_session(session_id):
    path = os.path.join(_sessions_dir(), _safe_session_file(session_id))
    if not os.path.exists(path):
        return jsonify({"status": "error", "message": "会话不存在"}), 404
    try:
        data = _read_archive_file(path)
    except Exception as e:
        return jsonify({"status": "error", "message": f"会话文件损坏: {e}"}), 500
    # 切换前先保存当前对话（不刷新其时间戳，避免列表顺序跳动）
    _save_current_session(touch=False)
    ai_brain.load_session(data)
    # 载入历史会话后抑制固定方案的自动 kickoff，避免刷新页面时被意外切走
    pending, _, input_run_id, _ = _refinement_kickoff()
    if os.path.exists(_refinement_input_path()):
        ai_brain.current_refinement_run_id = input_run_id
    return jsonify({
        "status": "success",
        "id": data.get("id", session_id),
        "messages": data.get("messages", []),
        "tokens": ai_brain.token_manager.get_status_report(),
    })

@app.route('/api/sessions/<session_id>/rename', methods=['POST'])
def rename_session(session_id):
    """重命名历史对话"""
    path = os.path.join(_sessions_dir(), _safe_session_file(session_id))
    if not os.path.exists(path):
        return jsonify({"status": "error", "message": "会话不存在"}), 404
    title = str((request.json or {}).get("title") or "").strip()
    if not title:
        return jsonify({"status": "error", "message": "名称不能为空"}), 400
    try:
        data = _read_archive_file(path)
        data["title"] = title[:80]
        data["title_custom"] = True  # 用户自定义名，自动保存不得覆盖
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        return jsonify({"status": "error", "message": f"重命名失败: {e}"}), 500
    return jsonify({"status": "success", "title": data["title"]})

@app.route('/api/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    path = os.path.join(_sessions_dir(), _safe_session_file(session_id))
    if os.path.exists(path):
        os.remove(path)
    if getattr(ai_brain, "current_session_id", None) == session_id:
        ai_brain.current_session_id = None
    return jsonify({"status": "success"})

@app.route('/shutdown', methods=['POST'])
def shutdown():
    print(">>> 收到系统关闭指令... 准备退出")
    _save_current_session()
    # 延迟 0.5 秒发送终止信号，确保前端能收到 HTTP 200 OK 响应
    threading.Timer(0.5, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
    return jsonify({"status": "ok"})

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')

    def _generate():
        try:
            for chunk in ai_brain.chat_yield(user_input):
                yield f"data: {chunk}\n\n"
        finally:
            # 对话流结束（或客户端中断）后自动保存当前会话
            _save_current_session()

    return Response(stream_with_context(_generate()), mimetype='text/event-stream')

@app.route('/approve', methods=['POST'])
def approve():
    data = request.json
    tid = data.get('tool_call_id')
    cmd = data.get('command')
    is_ok = data.get('approved')
    a_type = data.get('audit_type')
    
    # 核心修复：后台去唤醒线程，并直接给前端返回 JSON，不再使用 stream_with_context
    ai_brain.handle_approval(tid, cmd, is_ok, a_type)
    return jsonify({"status": "ok, thread resumed"})

# ================= 配置管理接口 =================
@app.route('/api/config', methods=['GET'])
def get_config():
    """获取当前配置，并扫描可用的工具和技能列表发给前端"""
    config_data = ai_brain.config_manager.config
    
    # 获取可用的工具列表
    available_tools = list(TOOL_REGISTRY.keys())
    
    # 获取可用的技能列表 (排除核心记忆文件)
    available_skills = []
    skills_dir = ai_brain.skill_manager.skill_dir
    if os.path.exists(skills_dir):
        available_skills = [f for f in os.listdir(skills_dir) 
                            if f.endswith(".md") and f not in ["base.md", "catalogue.md"]]
    
    return jsonify({
        "config": config_data,
        "available_tools": available_tools,
        "available_skills": available_skills,
        "token_status": ai_brain.token_manager.get_status_report(),
        "context_window": ai_brain.token_manager.context_window,
    })

@app.route('/api/config', methods=['POST'])
def save_config():
    """保存前端传来的新配置并热更新 AI 记忆"""
    new_config = request.json

    # Token 上限走专用通道：校验范围并持久化
    token_limit = new_config.pop("token_limit", None)
    if token_limit is not None:
        try:
            ai_brain.set_token_limit(int(token_limit))
        except (TypeError, ValueError):
            return jsonify({"status": "error", "message": "Token 上限必须是数字"}), 400

    ai_brain.config_manager.config.update(new_config)
    ai_brain.config_manager.save()

    ai_brain.tool_manager.reset()
    ai_brain.context_manager.refresh_base()

    return jsonify({
        "status": "success",
        "tokens": ai_brain.token_manager.get_status_report(),
        "tokens_used": ai_brain.token_manager.total_used,
        "tokens_limit": ai_brain.token_manager.max_threshold,
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True, use_reloader=False)
