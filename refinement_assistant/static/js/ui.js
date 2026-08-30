/* static/js/ui.js */
function toggleNightMode() {
    const body = document.body;
    const btn = document.getElementById('night-btn');
    
    body.classList.toggle('dark-mode');
    
    if (body.classList.contains('dark-mode')) {
        btn.innerText = '日间模式';
        localStorage.setItem('hypoforge_theme', 'dark');
    } else {
        btn.innerText = '夜间模式';
        localStorage.setItem('hypoforge_theme', 'light');
    }
}

function loadThemePreference() {
    const theme = localStorage.getItem('hypoforge_theme');
    if (theme === 'dark') {
        document.body.classList.add('dark-mode');
        const btn = document.getElementById('night-btn');
        if (btn) {
            btn.innerText = '日间模式';
        }
    }
}

async function openConfigModal() {
    try {
        const res = await fetch('/api/config');
        const data = await res.json();
        
        const config = data.config;
        const availableTools = data.available_tools;
        const availableSkills = data.available_skills;
        
        document.getElementById('config-allowlist').checked = config.enable_allowlist !== false;
        document.getElementById('config-full-trust').checked = config.full_trust_mode === true;

        const tokenLimitInput = document.getElementById('config-token-limit');
        tokenLimitInput.value = config.token_limit || 1000000;
        document.getElementById('config-token-status').innerText = data.token_status || '';
        document.getElementById('config-context-window').innerText =
            (data.context_window || 131072).toLocaleString();
        
        const toolsContainer = document.getElementById('tools-checkbox-container');
        toolsContainer.innerHTML = '';
        availableTools.forEach(tool => {
            const isChecked = !(config.disabled_tools || []).includes(tool);
            toolsContainer.innerHTML += `
                <label class="checkbox-item">
                    <input type="checkbox" value="${tool}" data-type="tool" ${isChecked ? 'checked' : ''}>
                    ${tool}
                </label>
            `;
        });

        const skillsContainer = document.getElementById('skills-checkbox-container');
        skillsContainer.innerHTML = '';
        if (availableSkills.length === 0) {
            skillsContainer.innerHTML = '<span style="color:#999;font-size:12px;padding:5px;">暂无额外的技能文件</span>';
        } else {
            availableSkills.forEach(skill => {
                const isChecked = !(config.disabled_skills || []).includes(skill);
                skillsContainer.innerHTML += `
                    <label class="checkbox-item" title="${skill}">
                        <input type="checkbox" value="${skill}" data-type="skill" ${isChecked ? 'checked' : ''}>
                        ${skill.length > 25 ? skill.substring(0, 22) + '...' : skill}
                    </label>
                `;
            });
        }
        
        document.getElementById('config-modal').style.display = 'block';
    } catch (e) {
        alert("获取配置失败，请确保后端正常运行。");
    }
}

function closeConfigModal() {
    document.getElementById('config-modal').style.display = 'none';
}

async function saveConfig() {
    const allowlist = document.getElementById('config-allowlist').checked;
    
    const disabledTools = [];
    document.querySelectorAll('#tools-checkbox-container input[type="checkbox"]').forEach(cb => {
        if (!cb.checked) {
            disabledTools.push(cb.value);
        }
    });

    const disabledSkills = [];
    document.querySelectorAll('#skills-checkbox-container input[type="checkbox"]').forEach(cb => {
        if (!cb.checked) {
            disabledSkills.push(cb.value);
        }
    });

    const newConfig = {
        enable_allowlist: allowlist,
        full_trust_mode: document.getElementById('config-full-trust').checked,
        disabled_tools: disabledTools,
        disabled_skills: disabledSkills
    };

    // 开启完全信任时二次确认
    if (newConfig.full_trust_mode && !confirm(
        '⚠️ 即将开启【完全信任模式】：AI 将可以不经确认直接创建/修改/删除文件并执行任意命令。\n\n确定开启吗？（可随时回到设置关闭）'
    )) {
        newConfig.full_trust_mode = false;
        document.getElementById('config-full-trust').checked = false;
    }

    // Token 上限：留空则不修改
    const tokenLimit = parseInt(document.getElementById('config-token-limit').value, 10);
    if (!isNaN(tokenLimit) && tokenLimit > 0) {
        newConfig.token_limit = tokenLimit;
    }

    try {
        const res = await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(newConfig)
        });
        const data = await res.json();
        if (!res.ok) {
            throw new Error(data.message || '保存失败');
        }

        closeConfigModal();

        const chatBox = document.getElementById('chat-box');
        chatBox.innerHTML += `<div class="status-tag">⚙️ [系统] 模块配置已保存，后台认知热更新完成。Token 状态: ${formatTokenChip(data.tokens_used, data.tokens_limit)}</div>`;
        document.getElementById('token-info').innerText =
            formatTokenChip(data.tokens_used, data.tokens_limit);
        chatBox.scrollTop = chatBox.scrollHeight;
    } catch (e) {
        alert("保存配置失败：" + e.message);
    }
}