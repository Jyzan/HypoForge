# Web UI Polish and Context Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the visual hierarchy of the existing M1–M6 interface and expose context/token savings without changing REST contracts or result drawers.

**Architecture:** Keep the mature inline application JavaScript intact, add a separately served polish stylesheet, and make small semantic HTML/JS additions for context metrics. `webapp.py` serves a strict asset whitelist from the existing static directory. The credential panel is collapsed by default so the scientific question remains the primary action.

**Tech Stack:** HTML5, CSS, vanilla JavaScript, Python `http.server`, in-app browser verification.

**Spec:** `docs/superpowers/specs/2026-08-23-hypoforge-continuous-optimization-design.md`

## Global Constraints

- Preserve all existing element IDs, REST endpoints, drawers, graph overlay and run actions.
- Never expose API key values in DOM summaries, events or persisted state.
- Serve only known static assets; reject path traversal and arbitrary file reads.
- Keep desktop and narrow-screen layouts usable.
- Do not push the branch.

---

### Task 1: Add safe static-asset delivery

**Files:**
- Modify: `hypoforge/webapp.py`
- Modify: `scripts/test_pipeline.py`
- Create: `hypoforge/web/ui-polish.css`

- [ ] **Step 1: Add failing HTTP tests**

Assert `/assets/ui-polish.css` returns 200 with `text/css`, a missing asset returns 404, and `/assets/../webapp.py` cannot read outside `hypoforge/web`.

- [ ] **Step 2: Verify RED**

Expected: asset request currently falls through to the HTML page.

- [ ] **Step 3: Add whitelist-based asset serving**

Map only `ui-polish.css` and future `app.js` to files under `static_dir`, with explicit MIME types, UTF-8/body length, no-store cache, and 404 for every other asset path.

- [ ] **Step 4: Run HTTP tests GREEN**

Expected: all asset/security cases PASS.

### Task 2: Improve page hierarchy and responsive styling

**Files:**
- Modify: `hypoforge/web/index.html`
- Create: `hypoforge/web/ui-polish.css`

- [ ] **Step 1: Link the external stylesheet and repair malformed inline CSS**

Remove the stray unmatched brace after the scrollbar rules and add `<link rel="stylesheet" href="/assets/ui-polish.css">` after the inline style so polish rules layer safely over existing behavior.

- [ ] **Step 2: Collapse credentials by default**

Remove the `open` attribute from `#credCard`; retain every field and its security note.

- [ ] **Step 3: Add visual polish**

Use a subtle laboratory-grid/radial background, translucent header/input bar, stronger question/input focus, compact hero, clearer module-state accents, improved cards and narrow-screen spacing. Respect `prefers-reduced-motion`.

- [ ] **Step 4: Verify CSS syntax and inline JavaScript syntax**

Parse/extract the inline script with the existing Node syntax check and confirm HTML IDs remain unique.

### Task 3: Show context budget and evidence-pack metrics

**Files:**
- Modify: `hypoforge/web/index.html`
- Modify: `scripts/test_pipeline.py`

- [ ] **Step 1: Add failing rendered-UI test**

Assert the page contains counters for context packs, estimated context tokens, included evidence and dropped-but-traceable evidence.

- [ ] **Step 2: Add semantic metric cards**

Extend `state-counters` with IDs `contextPackCount`, `contextTokenCount`, `contextIncludedCount`, and `contextDroppedCount`.

- [ ] **Step 3: Aggregate `llm_context_built` events**

In `updateStats`, sum manifest values from all events and render them with localized integer formatting. Reset naturally when `allEvents` resets.

- [ ] **Step 4: Run UI/HTTP and full tests**

Expected: all tests PASS.

### Task 4: Browser verification and local commit

**Files:**
- Test: running web server and rendered page

- [ ] **Step 1: Start an isolated server on an unused port**

- [ ] **Step 2: Inspect desktop and narrow viewports**

Verify empty state, collapsed credentials, input focus, run workspace, pipeline cards, context counters, drawer and graph overlay.

- [ ] **Step 3: Commit locally**

```powershell
git add hypoforge/webapp.py hypoforge/web/index.html hypoforge/web/ui-polish.css scripts/test_pipeline.py
git commit -m "feat(ui): polish pipeline and expose context budgets"
```
