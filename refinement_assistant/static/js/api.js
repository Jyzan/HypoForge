/* static/js/api.js */
window.availableMemories = [];
window.chatTranscript = [];

async function fetchMemories() {
    try {
        const res = await fetch('/api/memories');
        const data = await res.json();
        window.availableMemories = data.memories || [];
    } catch (e) {
        console.error('加载记忆列表失败', e);
    }
}

function formatTokenChip(used, limit) {
    const u = Number(used || 0).toLocaleString();
    const l = Number(limit || 0).toLocaleString();
    return `Token: ${u} / ${l}`;
}

async function syncStatus() {
    try {
        const res = await fetch('/init');
        const data = await res.json();
        document.getElementById('token-info').innerText =
            formatTokenChip(data.tokens_used, data.tokens_limit);
        document.getElementById('model-info').innerText = `模型: ${data.model || '未知'}`;
    } catch (e) {
        document.getElementById('token-info').innerText = 'Token: —';
    }
}

// ================= 对话条（当前对话标题 / token / 细化方案） =================
async function updateConvHeader() {
    try {
        const res = await fetch('/api/session/current');
        const data = await res.json();
        window.currentSessionId = data.session_id || null;
        document.getElementById('conv-title').innerText =
            (data.messages && data.messages.length) ? (data.title || '未命名对话') : '新对话';
        document.getElementById('token-info').innerText =
            formatTokenChip(data.tokens_used, data.tokens_limit);
    } catch (e) {
        console.error('更新对话条失败', e);
    }
}

async function renameCurrentSession() {
    const sid = window.currentSessionId;
    if (!sid) {
        alert('当前还没有可命名的对话（发送第一条消息后自动创建）。');
        return;
    }
    const current = document.getElementById('conv-title').innerText;
    const name = prompt('重命名当前对话：', current === '新对话' ? '' : current);
    if (name === null) return;
    if (!name.trim()) {
        alert('名称不能为空。');
        return;
    }
    await renameSession(sid, name.trim());
    updateConvHeader();
}

async function renameSession(id, title) {
    try {
        const res = await fetch(`/api/sessions/${encodeURIComponent(id)}/rename`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.message || '重命名失败');
        refreshSessions();
        return true;
    } catch (e) {
        alert(`重命名失败：${e.message}`);
        return false;
    }
}

// 页面刷新后恢复服务端当前对话的显示
async function restoreLiveSession() {
    try {
        const res = await fetch('/api/session/current');
        const data = await res.json();
        if (data.messages && data.messages.length) {
            window.chatTranscript = data.messages.map(m => ({
                role: m.role,
                content: m.content,
                created_at: null
            }));
            renderArchivedConversation(window.chatTranscript);
            document.getElementById('token-info').innerText =
                formatTokenChip(data.tokens_used, data.tokens_limit);
        }
        window.currentSessionId = data.session_id || null;
    } catch (e) {
        console.error('恢复当前对话失败', e);
    }
}
async function refreshSessions() {
    try {
        const res = await fetch('/api/sessions');
        const data = await res.json();
        renderSessionList(data.sessions || []);
    } catch (e) {
        console.error('加载历史对话失败', e);
    }
}

function renderSessionList(sessions) {
    const list = document.getElementById('session-list');
    if (!list) return;
    if (!sessions.length) {
        list.innerHTML = '<div class="session-empty">暂无历史对话。<br>每轮对话结束后会自动保存在这里。</div>';
        return;
    }
    list.innerHTML = sessions.map(s => `
        <div class="session-item" title="${escapeHtml(s.title)}" onclick="loadSessionById('${escapeHtml(s.id)}')">
            <div class="session-item-title">${escapeHtml(s.title)}</div>
            <div class="session-item-meta">${escapeHtml(s.updated_at || '')} · ${s.message_count || 0} 条</div>
            <button class="session-rename" title="重命名" onclick="event.stopPropagation(); renameSessionPrompt('${escapeHtml(s.id)}', '${escapeHtml(s.title).replace(/'/g, "\\'")}')">✎</button>
            <button class="session-delete" title="删除" onclick="event.stopPropagation(); deleteSessionById('${escapeHtml(s.id)}')">✕</button>
        </div>
    `).join('');
}

function renameSessionPrompt(id, currentTitle) {
    const name = prompt('重命名对话：', currentTitle);
    if (name === null || !name.trim()) return;
    renameSession(id, name.trim()).then(ok => {
        if (ok && window.currentSessionId === id) updateConvHeader();
    });
}

async function loadSessionById(id) {
    try {
        const res = await fetch(`/api/sessions/${encodeURIComponent(id)}/load`, { method: 'POST' });
        const data = await res.json();
        if (!res.ok) {
            throw new Error(data.message || '加载失败');
        }
        const messages = (data.messages || []).map(msg => ({
            role: msg.role,
            content: msg.content,
            created_at: null
        }));
        window.chatTranscript = messages.filter(m => m.content);
        renderArchivedConversation(window.chatTranscript);
        document.getElementById('token-info').innerText =
            formatTokenChip(data.tokens_used, data.tokens_limit);
        updateConvHeader();
        refreshSessions();
    } catch (e) {
        alert(`加载历史对话失败：${e.message}`);
    }
}

async function deleteSessionById(id) {
    if (!confirm('确定删除这条历史对话吗？删除后不可恢复。')) return;
    try {
        await fetch(`/api/sessions/${encodeURIComponent(id)}`, { method: 'DELETE' });
        refreshSessions();
    } catch (e) {
        alert(`删除失败：${e.message}`);
    }
}

// ================= 细化方案展示面板 =================
async function openRefinementPanel() {
    document.getElementById('refinement-modal').style.display = 'block';
    const detail = document.getElementById('refinement-detail');
    const meta = document.getElementById('refinement-meta');
    detail.innerHTML = '<div class="archive-empty">正在加载细化方案…</div>';
    meta.innerText = '';

    try {
        const res = await fetch('/api/refinement/output');
        const data = await res.json();
        if (!data.exists) {
            detail.innerHTML = `<div class="archive-empty">${escapeHtml(data.reason || '当前对话还没有细化方案产出。')}</div>`;
            meta.innerText = '';
            return;
        }
        meta.innerText = `run_id: ${data.run_id} · 保存于 ${data.updated_at}`;
        detail.innerHTML = `<div class="content-box refinement-content">${renderMarkdownWithMath(data.markdown || '')}</div>`;
        if (window.MathJax && window.MathJax.typesetPromise) {
            MathJax.typesetPromise([detail]).catch(err => console.log('MathJax error:', err));
        }
    } catch (e) {
        detail.innerHTML = `<div class="archive-empty">加载失败：${escapeHtml(e.message)}</div>`;
    }
}

function closeRefinementPanel() {
    document.getElementById('refinement-modal').style.display = 'none';
}

function downloadRefinementMd() {
    window.open('/api/refinement/output/download', '_blank');
}

async function resetChat() {
    try {
        const res = await fetch('/reset', { method: 'POST' });
        const data = await res.json();
        window.chatTranscript = [];
        document.getElementById('chat-box').innerHTML = '<div class="status-tag">已开启新对话。历史对话已自动保存，可从左侧查看。</div>';
        document.getElementById('token-info').innerText = 'Token: 0';
        updateConvHeader();
        refreshSessions();
    } catch (e) {
        alert('重置失败。');
    }
}

async function shutdownApp() {
    try {
        const res = await fetch('/shutdown', { method: 'POST' });
        if (res.ok) {
            document.body.innerHTML = `
                <div style="text-align:center; margin-top:100px; font-family:sans-serif;">
                    <h1 style="color:#c0392b;">工作台已关闭</h1>
                    <p>HypoForge 细化工作台已安全关闭。正在尝试关闭窗口...</p>
                </div>
            `;
            setTimeout(() => {
                window.close();
                window.location.href = 'about:blank';
            }, 800);
        }
    } catch (e) {
        alert('系统关闭遇到异常，可能是后台已强制退出。');
        window.close();
        window.location.href = 'about:blank';
    }
}

async function doubleToken() {
    const res = await fetch('/double_token', { method: 'POST' });
    const data = await res.json();
    document.getElementById('token-info').innerText = `Token 实时状态: ${data.tokens}`;
    alert('额度已翻倍。');
}

async function setTokenLimit(limit) {
    try {
        const res = await fetch('/set_token', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ limit: limit })
        });
        const data = await res.json();
        document.getElementById('token-info').innerText = `Token 实时状态: ${data.tokens}`;

        const chatBox = document.getElementById('chat-box');
        chatBox.innerHTML += `<div class="status-tag">[系统] Token 阈值已更新为 ${limit}。</div>`;
        chatBox.scrollTop = chatBox.scrollHeight;
    } catch (e) {
        alert('修改 Token 阈值失败。');
    }
}

function escapeHtml(text) {
    return String(text || '').replace(/[&<>"']/g, (ch) => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;'
    }[ch]));
}

function defaultArchiveTitle() {
    const firstUser = (window.chatTranscript || []).find(m => m.role === 'user');
    if (firstUser && firstUser.content) {
        return firstUser.content.slice(0, 32);
    }
    return new Date().toLocaleString();
}

async function saveCurrentArchive() {
    const messages = window.chatTranscript || [];
    if (messages.length === 0) {
        alert('当前还没有可存档的消息。');
        return;
    }

    const title = prompt('给这次聊天存档取个名字：', defaultArchiveTitle());
    if (title === null) return;

    try {
        const res = await fetch('/api/archives', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                title: title.trim() || defaultArchiveTitle(),
                messages
            })
        });
        const data = await res.json();
        if (!res.ok) {
            throw new Error(data.message || '保存失败');
        }
        alert(`存档成功：${data.archive.title}`);
    } catch (e) {
        alert(`存档失败：${e.message}`);
    }
}

async function openArchiveModal() {
    document.getElementById('archive-modal').style.display = 'block';
    const listBox = document.getElementById('archive-list');
    const detailBox = document.getElementById('archive-detail');
    listBox.innerHTML = '<div class="archive-empty">正在加载...</div>';
    detailBox.innerHTML = '选择一个存档查看内容。';

    try {
        const res = await fetch('/api/archives');
        const data = await res.json();
        const archives = data.archives || [];
        if (archives.length === 0) {
            listBox.innerHTML = '<div class="archive-empty">暂无存档。</div>';
            return;
        }

        listBox.innerHTML = archives.map(item => `
            <button class="archive-item" onclick="loadArchive('${escapeHtml(item.id)}')">
                <span class="archive-title">${escapeHtml(item.title)}</span>
                <span class="archive-meta">${escapeHtml(item.created_at || '')} · ${item.message_count || 0} 条</span>
            </button>
        `).join('');
    } catch (e) {
        listBox.innerHTML = `<div class="archive-empty">加载失败：${escapeHtml(e.message)}</div>`;
    }
}

function closeArchiveModal() {
    document.getElementById('archive-modal').style.display = 'none';
}

async function loadArchive(id) {
    const detailBox = document.getElementById('archive-detail');
    detailBox.innerHTML = '正在读取存档...';

    try {
        const res = await fetch(`/api/archives/${encodeURIComponent(id)}`);
        const archive = await res.json();
        if (!res.ok) {
            throw new Error(archive.message || '读取失败');
        }

        const messagesHtml = (archive.messages || []).map(msg => `
            <div class="archive-message archive-${escapeHtml(msg.role)}">
                <div class="archive-role">${msg.role === 'user' ? '用户' : '助手'}</div>
                <div class="archive-content">${escapeHtml(msg.content)}</div>
            </div>
        `).join('');

        detailBox.innerHTML = `
            <div class="archive-detail-title">${escapeHtml(archive.title)}</div>
            <div class="archive-detail-meta">${escapeHtml(archive.created_at || '')}</div>
            <div class="archive-actions">
                <button class="btn btn-archive" onclick="resumeArchive('${escapeHtml(archive.id || id)}')">从此继续</button>
            </div>
            <div class="archive-messages">${messagesHtml || '<div class="archive-empty">这个存档没有消息。</div>'}</div>
        `;
    } catch (e) {
        detailBox.innerHTML = `<div class="archive-empty">读取失败：${escapeHtml(e.message)}</div>`;
    }
}

async function resumeArchive(id) {
    try {
        const res = await fetch(`/api/archives/${encodeURIComponent(id)}/resume`, {
            method: 'POST'
        });
        const data = await res.json();
        if (!res.ok) {
            throw new Error(data.message || '恢复失败');
        }

        const archive = data.archive;
        const messages = (archive.messages || []).filter(msg =>
            (msg.role === 'user' || msg.role === 'assistant') && msg.content
        );
        window.chatTranscript = messages.map(msg => ({
            role: msg.role,
            content: msg.content,
            created_at: msg.created_at || new Date().toISOString()
        }));

        renderArchivedConversation(window.chatTranscript);
        document.getElementById('token-info').innerText = `Token 实时状态: ${data.tokens}`;
        closeArchiveModal();
    } catch (e) {
        alert(`恢复失败：${e.message}`);
    }
}

function renderArchivedConversation(messages) {
    const chatBox = document.getElementById('chat-box');
    chatBox.innerHTML = '<div class="status-tag">已从消息存档恢复对话，可以继续发送新消息。</div>';

    messages.forEach(msg => {
        const div = document.createElement('div');
        div.className = `msg ${msg.role === 'user' ? 'user' : 'ai'}`;
        if (msg.role === 'assistant') {
            div.innerHTML = `<div class="content-box">${renderMarkdownWithMath(msg.content)}</div>`;
        } else {
            div.textContent = msg.content;
        }
        chatBox.appendChild(div);
    });

    chatBox.scrollTop = chatBox.scrollHeight;
    if (window.MathJax && window.MathJax.typesetPromise) {
        MathJax.typesetPromise([chatBox]).catch(err => console.log('MathJax error:', err));
    }
}
