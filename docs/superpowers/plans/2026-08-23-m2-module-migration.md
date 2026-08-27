# M2 Module Namespace Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the complete M2 implementation beside M1–M6 under `hypoforge.modules.m2_literature` without breaking existing `hypoforge.literature` consumers or strict adapter selection.

**Architecture:** The new modules namespace is canonical. The old package becomes a single compatibility loader that aliases canonical package and submodule objects in `sys.modules`, so old and new imports resolve to identical classes rather than duplicated implementations. The stable registry facade remains `hypoforge.modules.m2_literature_search`.

**Tech Stack:** Python package imports, importlib/pkgutil compatibility aliases, pytest, git-aware file moves.

**Spec:** `docs/superpowers/specs/2026-08-23-hypoforge-continuous-optimization-design.md`

## Global Constraints

- Physically move every M2 implementation file; the legacy directory may contain only its compatibility `__init__.py`.
- Preserve old deep imports such as `hypoforge.literature.search.ranking`.
- Old and new imports of the same class/module must have object identity.
- Keep `hypoforge.modules.m2_literature_search` as the pipeline registry facade.
- Do not change public DTOs in `state.py` or search/reading behavior.
- Do not push the branch.

---

### Task 1: Lock the canonical and compatibility import contracts

**Files:**
- Modify: `scripts/test_pipeline.py`
- Test: `scripts/test_pipeline.py`

**Interfaces:**
- Produces: canonical imports from `hypoforge.modules.m2_literature`.
- Preserves: legacy imports from `hypoforge.literature` and every tested deep module.

- [ ] **Step 1: Add a failing canonical-package test**

Import `AgenticM2Adapter`, `SearchQuery`, `PaperRanker`, `FullTextReadingWorkflow`, and `OpenAlexSource` from both namespaces. Assert each old object `is` its new counterpart and that their `__module__` begins with `hypoforge.modules.m2_literature`.

- [ ] **Step 2: Verify RED**

Run:

```powershell
python -m pytest -q scripts/test_pipeline.py -k "m2_canonical_and_legacy_imports_share_identity"
```

Expected: FAIL because `hypoforge.modules.m2_literature` does not exist.

### Task 2: Move implementation and repair package-relative imports

**Files:**
- Move: `hypoforge/literature/**` → `hypoforge/modules/m2_literature/**`
- Create: `hypoforge/literature/__init__.py`
- Modify: moved Python modules whose root-relative import depth changes

**Interfaces:**
- Produces: canonical package `hypoforge.modules.m2_literature`.
- Produces: legacy loader that maps `hypoforge.literature[.*]` to canonical module objects.

- [ ] **Step 1: Move the directory with Git history preserved**

Run:

```powershell
git mv hypoforge/literature hypoforge/modules/m2_literature
```

- [ ] **Step 2: Repair imports that target the root `hypoforge` package**

For files directly inside `m2_literature`, change root imports from `..state`, `..tools`, `..observability`, `..protocol`, `..memory`, `..task_alignment`, and `..entity_normalization` to three-dot imports. For `search`, `reading`, and `sources` children, change root-targeting three-dot imports to four-dot imports. Leave imports such as `.models` and `..models` unchanged.

- [ ] **Step 3: Replace internal absolute legacy imports**

Change every `from hypoforge.literature...` inside the moved implementation to `from hypoforge.modules.m2_literature...`.

- [ ] **Step 4: Add the legacy alias loader**

Create `hypoforge/literature/__init__.py`. Import the canonical package, enumerate all canonical submodules with `pkgutil.walk_packages`, import them once, then register aliases in `sys.modules` by replacing the canonical prefix with `hypoforge.literature`. Re-export the canonical package's public attributes and set the legacy package entry to the canonical module object.

- [ ] **Step 5: Verify import identity GREEN**

Run the Task 1 test and a compile check. Expected: PASS and no duplicate class objects.

### Task 3: Switch production consumers and strict selection to canonical paths

**Files:**
- Modify: `hypoforge/modules/m2_literature_search.py`
- Modify: `hypoforge/strict_contracts.py`
- Modify: `hypoforge/webapp.py`
- Modify: `hypoforge/registry.py`
- Modify: `hypoforge/modules/m2_literature/adapter.py`
- Test: `scripts/test_pipeline.py`

**Interfaces:**
- Produces: `AgenticM2Module.strict_contract_family = "agentic_m2"`.
- Consumes: marker-based strict adapter selection in `ModuleRegistry`.

- [ ] **Step 1: Add a failing marker-selection test**

Assert that the M2 facade class carries `strict_contract_family == "agentic_m2"` and that Registry strict mode still returns `StrictM2LiteratureSearch` without comparing `__module__` strings.

- [ ] **Step 2: Verify RED**

Expected: FAIL because current strict selection relies on exact module path and class name.

- [ ] **Step 3: Switch production imports to canonical paths**

Update the facade, strict contracts, and web credential warning import. Update strict-contract compatibility rebinding to target the canonical adapter module.

- [ ] **Step 4: Replace path-string detection with the stable marker**

Set the marker on the facade/agentic module and make the Registry branch depend on `getattr(module_cls, "strict_contract_family", "") == "agentic_m2"`.

- [ ] **Step 5: Run canonical, legacy, registry and strict tests**

Expected: all selected tests PASS.

### Task 4: Verify the physical and runtime migration

**Files:**
- Modify: `hypoforge/modules/m2_literature/README.md`
- Test: full repository suite

**Interfaces:**
- Produces: documented canonical import path and compatibility policy.

- [ ] **Step 1: Update package documentation**

Document `hypoforge.modules.m2_literature` as canonical and `hypoforge.literature` as a temporary compatibility alias.

- [ ] **Step 2: Confirm the old directory contains no implementation**

Run:

```powershell
rg --files hypoforge/literature
```

Expected: only `hypoforge/literature/__init__.py`.

- [ ] **Step 3: Run compile and full tests**

```powershell
python -m compileall -q hypoforge
python -m pytest -q
```

Expected: all tests PASS.

- [ ] **Step 4: Run a standalone M2 construction smoke test**

Load `configs/web_ui.yaml`, build modules through `ModuleRegistry`, and assert the resulting M2 facade owns a canonical `AgenticM2Module` implementation.

- [ ] **Step 5: Commit locally**

```powershell
git add -A hypoforge scripts/test_pipeline.py
git commit -m "refactor(m2): move implementation beside pipeline modules"
```
