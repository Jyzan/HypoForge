/* static/js/chat.js */

// ================= 生成中保护（防误刷新） =================
window._streamActive = false;

function setStreamIndicator(on) {
    let el = document.getElementById('stream-indicator');
    if (on && !el) {
        el = document.createElement('div');
        el.id = 'stream-indicator';
        el.innerText = '⏳ 生成中，请勿刷新页面…';
        document.getElementById('conv-header').appendChild(el);
    } else if (!on && el) {
        el.remove();
    }
}

window.addEventListener('beforeunload', (e) => {
    if (window._streamActive) {
        e.preventDefault();
        e.returnValue = '正在生成中，刷新会中断本次生成。确定离开？';
        return e.returnValue;
    }
});

// 初始化
window.onload = () => {
    syncStatus();
    loadThemePreference();
    fetchMemories();
    refreshSessions();
    restoreLiveSession().then(() => {
        updateConvHeader();
        checkRefinementKickoff();
    });
};

// ================= 固定方案自动载入 =================
async function checkRefinementKickoff() {
    try {
        const res = await fetch('/api/refinement/kickoff');
        const data = await res.json();
        if (data.pending && data.message) {
            const box = document.getElementById('chat-box');
            const suffix = data.switch ? '（将自动保存当前对话并切换）' : '';
            box.innerHTML += `<div class="status-tag">🔗 已从 HypoForge 载入固定方案（run_id: ${escapeHtml(data.run_id || '')}）${suffix}，自动开始细化…</div>`;
            box.scrollTop = box.scrollHeight;
            if (data.switch) {
                await resetChat();  // 内部会把当前对话存入历史
            }
            await fetch('/api/refinement/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ run_id: data.run_id })
            });
            send(data.message);
        }
    } catch (e) {
        console.error('kickoff 检查失败', e);
    }
}

// ================= 输入框自适应高度 =================
function autoResizeInput() {
    const el = document.getElementById('user-input');
    if (!el || el.tagName !== 'TEXTAREA') return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 150) + 'px';
}
document.getElementById('user-input').addEventListener('input', autoResizeInput);

// ================= 智能指令自动补全 =================
const COMMAND_LIST = ['/new', '/end', '/token ', '/organize ', '/schedule', '/skill', '/summary', '/memory '];
const MEMORY_MODES = ['load ', 'save '];

document.getElementById('user-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        autoResizeInput();
        send();
    } else if (e.key === 'Tab') {
        e.preventDefault(); 
        const input = e.target;
        const text = input.value;
        const lowerText = text.toLowerCase();

        if (lowerText.startsWith('/memory ')) {
            const parts = text.split(' ');
            if (parts.length === 2) {
                const modeInput = parts[1].toLowerCase();
                const modeMatches = MEMORY_MODES.filter(m => m.startsWith(modeInput));
                if (modeMatches.length > 0) {
                    input.value = '/memory ' + modeMatches[0];
                }
                return; 
            }
            if (parts.length === 3 && lowerText.startsWith('/memory load ')) {
                const nameInput = parts[2].toLowerCase();
                const nameMatches = window.availableMemories.filter(m => m.toLowerCase().startsWith(nameInput));
                if (nameMatches.length > 0) {
                    input.value = `/memory load ${nameMatches[0]}`;
                }
                return; 
            }
        }

        if (lowerText.startsWith('/')) {
            const matches = COMMAND_LIST.filter(cmd => cmd.startsWith(lowerText));
            if (matches.length > 0) {
                input.value = matches[0];
            }
        }
    }
});

// ================= 权限审批卡片 =================
function renderApprovalCard(container, data) {
    const card = document.createElement('div');
    card.className = 'approval-card';

    const tool = escapeHtml(String(data.tool_name || '工具'));
    const message = data.message ? `<div class="approval-msg">${escapeHtml(String(data.message))}</div>` : '';
    const cmd = escapeHtml(String(data.command || ''));
    const cmdShort = cmd.length > 500 ? cmd.slice(0, 500) + `\n…（其余 ${cmd.length - 500} 字符已省略）` : cmd;

    card.innerHTML = `
        <div class="approval-title">⚠️ 权限申请：<b>${tool}</b></div>
        ${message}
        <pre class="approval-cmd">${cmdShort}</pre>
        <div class="approval-actions">
            <button class="btn btn-stop ap-deny">拒绝</button>
            <button class="btn btn-reset ap-approve">批准</button>
            <span class="approval-status"></span>
        </div>`;

    const decide = async (approved) => {
        const okBtn = card.querySelector('.ap-approve');
        const noBtn = card.querySelector('.ap-deny');
        okBtn.disabled = true;
        noBtn.disabled = true;
        card.classList.add(approved ? 'ap-approved' : 'ap-denied');
        card.querySelector('.approval-status').innerText = approved ? '✓ 已批准' : '✕ 已拒绝';
        try {
            await fetch('/approve', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    tool_call_id: data.tool_call_id,
                    command: data.command,
                    approved,
                    audit_type: data.audit_type
                })
            });
        } catch (e) {
            card.querySelector('.approval-status').innerText = '提交失败：' + e.message;
        }
    };
    card.querySelector('.ap-approve').onclick = () => decide(true);
    card.querySelector('.ap-deny').onclick = () => decide(false);

    container.appendChild(card);
}

// ================= 数学公式保护 =================
// marked 会把 LaTeX 中的 _ ^ 等符号当作 Markdown 语法吃掉，导致 MathJax 拿到损坏的公式
// 因此在 marked 解析前用占位符保护数学块，解析后再还原
function renderMarkdownWithMath(text) {
    const mathBlocks = [];

    // 先保护 $$...$$ 显示公式
    text = text.replace(/\$\$([\s\S]*?)\$\$/g, (_, math) => {
        mathBlocks.push('$$' + math + '$$');
        return `\x00MATH${mathBlocks.length - 1}\x00`;
    });

    // 再保护 $...$ 行内公式（不跨行）
    text = text.replace(/\$([^$\n]+?)\$/g, (_, math) => {
        mathBlocks.push('$' + math + '$');
        return `\x00MATH${mathBlocks.length - 1}\x00`;
    });

    let html = marked.parse(text);

    // 还原数学块
    html = html.replace(/\x00MATH(\d+)\x00/g, (_, id) => mathBlocks[parseInt(id)]);

    return html;
}

// ================= 核心流式渲染逻辑 =================
async function executeOrganize(item, details) {
    const box = document.getElementById('chat-box');
    const aiDiv = document.createElement('div');
    aiDiv.className = 'msg ai'; 
    aiDiv.innerHTML = '<div class="status-box"></div><div class="content-box">正在整理...</div>';
    box.appendChild(aiDiv);
    box.scrollTop = box.scrollHeight;

    try {
        const response = await fetch('/api/organize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ item: item, details: details })
        });
        await processStream(response, aiDiv.querySelector('.status-box'), aiDiv.querySelector('.content-box'));
    } catch (e) {
        aiDiv.querySelector('.content-box').innerText = "❌ 整理请求出错。";
    }
}

async function executeMemory(mode, name) {
    const box = document.getElementById('chat-box');
    const aiDiv = document.createElement('div');
    aiDiv.className = 'msg ai'; 
    aiDiv.innerHTML = `<div class="status-box"></div><div class="content-box">正在${mode === 'save' ? '封存' : '挂载'}记忆 [${name}.md]...</div>`;
    box.appendChild(aiDiv);
    box.scrollTop = box.scrollHeight;

    try {
        const response = await fetch('/api/memory', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode: mode, name: name })
        });

        await processStream(response, aiDiv.querySelector('.status-box'), aiDiv.querySelector('.content-box'));
        if (mode === 'save') fetchMemories(); 
    } catch (e) {
        aiDiv.querySelector('.content-box').innerText = "❌ 记忆处理请求出错，连接中断。";
    }
}

async function processStream(response, statusBox, contentBox) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    const chatBox = document.getElementById('chat-box');

    // SSE 事件可能被网络分包切开（审批事件携带完整文件内容，体积大），
    // 必须跨 chunk 缓冲、按 \n\n 切分；单条解析失败跳过而非中断整个流
    let buffer = '';
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let sep;
        while ((sep = buffer.indexOf('\n\n')) !== -1) {
            const line = buffer.slice(0, sep).trim();
            buffer = buffer.slice(sep + 2);
            if (!line.startsWith('data: ')) continue;

            let data;
            try {
                data = JSON.parse(line.substring(6));
            } catch (err) {
                console.warn('[SSE] 跳过无法解析的事件片段:', line.slice(0, 80));
                continue;
            }

            try {
                handleStreamEvent(data, statusBox, contentBox, chatBox);
            } catch (err) {
                console.error('[SSE] 事件处理异常:', err);
            }
        }
        chatBox.scrollTop = chatBox.scrollHeight;
    }
}

function handleStreamEvent(data, statusBox, contentBox, chatBox) {
    if (contentBox.innerText === "正在思考..." ||
        contentBox.innerText.startsWith("正在整理") ||
        contentBox.innerText.startsWith("正在封存") ||
        contentBox.innerText.startsWith("正在挂载")) {
        contentBox.innerText = "";
    }
    if (data.type === 'status') {
        // insertAdjacentHTML 而非 innerHTML +=：后者会重建子元素，
        // 销毁已渲染审批卡片上的按钮事件处理器
        statusBox.insertAdjacentHTML('beforeend', `<div class="status-tag">⚡ ${escapeHtml(String(data.content || ''))}</div>`);
        if (String(data.content || '').includes('Refinement saved')) {
            statusBox.insertAdjacentHTML('beforeend', `<div class="status-tag">📄 细化方案已保存，点顶栏「细化方案」查看与下载</div>`);
        }
        chatBox.scrollTop = chatBox.scrollHeight;
    } else if (data.type === 'mobile_progress') {
        // 执行子代理的委派/反馈信息，逐行展示
        const lines = String(data.content || '').split('\n').filter(l => l.trim());
        statusBox.insertAdjacentHTML('beforeend', lines.map(l => `<div class="status-tag">🛠 ${escapeHtml(l)}</div>`).join(''));
        chatBox.scrollTop = chatBox.scrollHeight;
    } else if (data.type === 'approval') {
        // 页面内审批卡片：不会像原生弹窗一样被浏览器拦截或堆叠
        renderApprovalCard(statusBox, data);
        chatBox.scrollTop = chatBox.scrollHeight;
    } else if (data.type === 'final') {
        contentBox.innerHTML = renderMarkdownWithMath(data.content);
        window.chatTranscript.push({
            role: 'assistant',
            content: data.content,
            created_at: new Date().toISOString()
        });
        const tk = parseTokenStatus(data.tokens);
        document.getElementById('token-info').innerText = formatTokenChip(tk.used, tk.limit);
        if (window.MathJax && window.MathJax.typesetPromise) {
            MathJax.typesetPromise([contentBox]).catch(err => console.log('MathJax error:', err));
        }
    } else if (data.type === 'error') {
        contentBox.innerHTML = `<span style="color:#e74c3c; font-weight:bold;">❌ 系统提示：${escapeHtml(String(data.content || ''))}</span>`;
        window.chatTranscript.push({
            role: 'assistant',
            content: `[ERROR] ${data.content}`,
            created_at: new Date().toISOString()
        });
        if (String(data.content || '').includes("Token")) {
            contentBox.innerHTML += `<br><span style="font-size: 13px; color: #7f8c8d;">💡 提示：您可以在设置中调大 Token 上限，或使用 <b>/new</b> 开启新对话清理上下文。</span>`;
        }
    }
}

function parseTokenStatus(text) {
    const m = /\[(\d+)\s*\/\s*(\d+)\]/.exec(String(text || ''));
    return m ? { used: Number(m[1]), limit: Number(m[2]) } : { used: 0, limit: 0 };
}

async function send(forcedText) {
    const fromInput = typeof forcedText !== 'string';
    const input = document.getElementById('user-input');
    const box = document.getElementById('chat-box');
    const text = (fromInput ? input.value : forcedText).trim();
    if (!text) return;

    if (handleCommand(text)) {
        if (fromInput) input.value = '';
        autoResizeInput();
        return;
    }

    box.innerHTML += `<div class="msg user">${escapeHtml(text)}</div>`;
    window.chatTranscript.push({
        role: 'user',
        content: text,
        created_at: new Date().toISOString()
    });
    if (fromInput) input.value = '';
    autoResizeInput();

    const aiDiv = document.createElement('div');
    aiDiv.className = 'msg ai';
    aiDiv.innerHTML = '<div class="status-box"></div><div class="content-box">正在思考...</div>';
    box.appendChild(aiDiv);
    box.scrollTop = box.scrollHeight;

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text })
        });
        window._streamActive = true;
        setStreamIndicator(true);
        await processStream(response, aiDiv.querySelector('.status-box'), aiDiv.querySelector('.content-box'));
        // 对话流结束后后端已自动保存会话，刷新左侧列表与对话条（标题/token）
        refreshSessions();
        updateConvHeader();
    } catch (e) {
        aiDiv.querySelector('.content-box').innerHTML = `<span style="color:red">连接中断。</span>`;
    } finally {
        window._streamActive = false;
        setStreamIndicator(false);
    }
}
