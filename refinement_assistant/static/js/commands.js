/* static/js/commands.js */
function handleCommand(text) {
    const raw = text.trim();
    if (!raw.startsWith('/')) return false;

    const parts = raw.split(' ');
    const cmd = parts[0].toLowerCase();

    switch (cmd) {
        case '/new':
            resetChat();
            return true;
        case '/end':
            if (confirm("确定要通过指令关闭系统吗？")) {
                shutdownApp();
            }
            return true;
        case '/token':
            if (parts.length > 1) {
                const newLimit = parseInt(parts[1]);
                if (!isNaN(newLimit)) {
                    setTokenLimit(newLimit);
                } else {
                    alert("指令格式错误，请输入有效的数字，例如：/token 5000");
                }
            } else {
                alert("请输入要设置的 Token 阈值，例如：/token 5000");
            }
            return true;
        case '/organize':
            if (parts.length > 1) {
                const item = parts[1].toLowerCase();
                const details = parts.slice(2).join(' '); 
                executeOrganize(item, details);
            } else {
                alert("请输入要整理的项目，例如：/organize skills 不要动workspace.md");
            }
            return true;
        case '/schedule':
            openScheduleModal();
            return true;
        case '/memory':
            if (parts.length === 3) {
                const mode = parts[1].toLowerCase();
                const name = parts[2];
                if (mode === 'save' || mode === 'load') {
                    executeMemory(mode, name);
                } else {
                    alert("指令格式错误！mode 只能是 save 或 load。");
                }
            } else {
                alert("指令格式错误！\n正确格式：/memory [mode] [name]\n示例：/memory save protein_case");
            }
            return true;
        case '/skill':
        case '/summary':
            alert(`指令 [${cmd}] 属于进阶功能，目前正在研发中...`);
            return true;
        default:
            return false;
    }
}
