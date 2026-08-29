async function openScheduleModal() {
    document.getElementById('schedule-modal').style.display = 'block';
    await loadSchedule();
}

function closeScheduleModal() {
    document.getElementById('schedule-modal').style.display = 'none';
}

async function loadSchedule() {
    const grid = document.getElementById('schedule-grid');
    grid.innerHTML = '<div class="schedule-empty">正在加载时间表...</div>';

    try {
        const res = await fetch('/api/schedule');
        const schedule = await res.json();
        const days = schedule.days || [];
        if (days.length === 0) {
            grid.innerHTML = '<div class="schedule-empty">未来一周还没有安排。</div>';
            return;
        }

        grid.innerHTML = days.map(day => {
            const items = day.items || [];
            const itemHtml = items.length
                ? items.map(item => `
                    <div class="schedule-item priority-${escapeHtml(item.priority || 'medium')}">
                        <div class="schedule-time">${escapeHtml(item.time || '未定时间')}</div>
                        <div class="schedule-task">
                            <strong>${escapeHtml(item.project || '未命名项目')}</strong>
                            <span>${escapeHtml(item.task || '')}</span>
                        </div>
                        ${item.notes ? `<div class="schedule-notes">${escapeHtml(item.notes)}</div>` : ''}
                    </div>
                `).join('')
                : '<div class="schedule-empty">暂无安排</div>';

            return `
                <section class="schedule-day">
                    <div class="schedule-day-head">
                        <span>${escapeHtml(day.weekday || '')}</span>
                        <strong>${escapeHtml(day.date || '')}</strong>
                    </div>
                    <div class="schedule-items">${itemHtml}</div>
                </section>
            `;
        }).join('');
    } catch (e) {
        grid.innerHTML = `<div class="schedule-empty">加载失败：${escapeHtml(e.message)}</div>`;
    }
}
