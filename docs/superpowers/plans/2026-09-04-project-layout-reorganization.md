# Project Layout Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize scripts, configuration, data, generated artifacts, and documentation into responsibility-based directories without changing model or analysis behavior.

**Architecture:** Keep the model/CLI module intact under `src/casualvae`, group operational entry points under `scripts`, store configuration under `configs`, separate derived data under `data`, and place models/results under `artifacts`. Resolve defaults from each script's repository root so commands work independently of the caller's current directory.

**Tech Stack:** Python 3.10 from conda environment `casualvae_01`, pathlib, unittest, PowerShell, Git.

**Spec:** `docs/superpowers/specs/2026-09-04-project-layout-reorganization-design.md`

## Global Constraints

- Do not change model architecture, losses, zero-mode behavior, SCM behavior, training algorithms, or statistical formulas.
- Preserve every dataset, trained model, CSV, JSON, history, and analysis artifact; only disposable caches and the Word lock file may be deleted.
- Use `casualvae_01` for every Python command.
- Do not overwrite an existing destination.
- Resolve and validate absolute source and destination paths before directory moves.
- Preserve unrelated working-tree changes.

---

### Task 1: Lock the target layout and baseline

**Files:**
- Create: `tests/test_project_layout.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: target mapping from the design spec.
- Produces: executable layout assertions and ignore rules for `data/` and `artifacts/`.

- [ ] **Step 1: Record the pre-migration behavioral baseline**

Run:

```powershell
conda run -n casualvae_01 python -m unittest discover -s tests -v
Import-Csv outputs/multiseed_analysis/stability_comparison.csv | Format-Table
```

Expected: 30 tests pass and the comparison contains both variable- and fixed-split rows.

- [ ] **Step 2: Write the failing layout test**

```python
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

class ProjectLayoutTests(unittest.TestCase):
    def test_canonical_project_paths_exist(self):
        expected = [
            ROOT / "src/casualvae/module_aclf_partial_multimodal.py",
            ROOT / "scripts/training/run_multiseed.py",
            ROOT / "scripts/analysis/analyze_multiseed.py",
            ROOT / "configs/aclf_full_multimodal_config.json",
            ROOT / "data/structured_full",
            ROOT / "artifacts/multiseed/fixed_split",
            ROOT / "artifacts/analysis/multiseed/stability_comparison.csv",
        ]
        self.assertEqual([str(path) for path in expected if not path.exists()], [])

    def test_obsolete_root_locations_are_absent(self):
        obsolete = [ROOT / "module_aclf_partial_multimodal.py", ROOT / "outputs"]
        self.assertEqual([str(path) for path in obsolete if path.exists()], [])
```

- [ ] **Step 3: Run the layout test and verify RED**

Run:

```powershell
conda run -n casualvae_01 python -m unittest tests.test_project_layout -v
```

Expected: FAIL because canonical destinations do not exist and old locations still exist.

- [ ] **Step 4: Add destination ignore rules**

Append these rules while retaining the legacy `outputs/` rule:

```gitignore
data/
artifacts/
```

- [ ] **Step 5: Verify Git will ignore relocated generated files**

Run:

```powershell
git check-ignore data/example.csv artifacts/example/model.pt
```

Expected: both paths are reported as ignored.

---

### Task 2: Move source, scripts, configuration, and documentation

**Files:**
- Move: `module_aclf_partial_multimodal.py` → `src/casualvae/module_aclf_partial_multimodal.py`
- Move: root preprocessing scripts → `scripts/preprocessing/`
- Move: `run_multiseed.py` → `scripts/training/run_multiseed.py`
- Move: `analyze_multiseed.py` → `scripts/analysis/analyze_multiseed.py`
- Move: `extract.py` → `scripts/utilities/extract.py`
- Move: `aclf_full_multimodal_config.json` → `configs/aclf_full_multimodal_config.json`
- Move: `Technical_Proposal_V4.docx` → `docs/Technical_Proposal_V4.docx`
- Move: `模型架构.png` → `docs/assets/模型架构.png`
- Create: `src/casualvae/__init__.py`
- Create: package markers below `scripts/`

**Interfaces:**
- Consumes: existing files without content transformation.
- Produces: canonical code/config/docs tree used by path updates and tests.

- [ ] **Step 1: Create destination directories**

Create exactly `src/casualvae`, `scripts/preprocessing`, `scripts/training`,
`scripts/analysis`, `scripts/utilities`, `configs`, and `docs/assets` after validating
they resolve below the repository root.

- [ ] **Step 2: Move tracked files with Git and untracked files with PowerShell**

Use `git mv` for tracked files and `Move-Item -LiteralPath` for untracked files. Before
each group, reject any existing destination. Do not move `readme.md`, `requirements.txt`,
`.gitignore`, `tests`, or `.git`.

- [ ] **Step 3: Add package markers**

Create empty `__init__.py` files in `src/casualvae`, `scripts`, and each script category.

- [ ] **Step 4: Confirm file counts and names**

Run `rg --files src scripts configs docs` and compare every mapping against the spec.

---

### Task 3: Move data and generated artifacts

**Files:**
- Move: `outputs/augmented` → `data/augmented`
- Move: `outputs/structured_full` → `data/structured_full`
- Move: `outputs/run_*` → `artifacts/runs/run_*`
- Move: `outputs/multiseed` → `artifacts/multiseed/variable_split`
- Move: `outputs/multiseed_fixed_split` → `artifacts/multiseed/fixed_split`
- Move: `outputs/multiseed_analysis` → `artifacts/analysis/multiseed`

**Interfaces:**
- Consumes: 3.8 GB of existing generated files.
- Produces: separated data, training artifact, and analysis trees without content changes.

- [ ] **Step 1: Validate every directory move**

Resolve each source, assert it begins with the repository root, construct each absolute
destination below the repository, and assert the destination does not exist.

- [ ] **Step 2: Create parent directories**

Create `data`, `artifacts/runs`, `artifacts/multiseed`, and `artifacts/analysis`.

- [ ] **Step 3: Move each source directory as one same-volume operation**

Use native PowerShell `Move-Item -LiteralPath` for each validated source. Do not copy or
delete source data separately.

- [ ] **Step 4: Remove the empty legacy output directory**

Remove `outputs` only after confirming it is empty. If any unexpected file remains,
stop and report it instead of deleting it.

- [ ] **Step 5: Run the layout test and verify the filesystem portion is GREEN**

Run:

```powershell
conda run -n casualvae_01 python -m unittest tests.test_project_layout -v
```

Expected: canonical paths exist and obsolete root locations are absent.

---

### Task 4: Update imports and path defaults

**Files:**
- Modify: `scripts/preprocessing/*.py`
- Modify: `scripts/training/run_multiseed.py`
- Modify: `scripts/analysis/analyze_multiseed.py`
- Modify: `tests/*.py`

**Interfaces:**
- Consumes: repository root derived as `Path(__file__).resolve().parents[2]` for scripts.
- Produces: CLIs that address `data`, `configs`, `src`, and `artifacts` independent of CWD.

- [ ] **Step 1: Update test imports to the canonical module**

Insert the repository `src` directory into `sys.path` in model tests and import:

```python
from casualvae.module_aclf_partial_multimodal import ...
```

Import script tests with:

```python
from scripts.analysis import analyze_multiseed
from scripts.training import run_multiseed
```

- [ ] **Step 2: Run the suite and verify import failures**

Run:

```powershell
conda run -n casualvae_01 python -m unittest discover -s tests -v
```

Expected: tests reach the relocated modules; path-default tests still fail until the
entry-point defaults are updated.

- [ ] **Step 3: Update preprocessing defaults**

For each preprocessing script, derive the repository root from `__file__` and replace
machine-specific absolute paths with `data/augmented`, `data/structured_full`, and
`configs/aclf_full_multimodal_config.json` as applicable. Preserve all transformation
logic.

- [ ] **Step 4: Update training runner defaults and child script path**

Make the runner use absolute canonical paths for:

```text
src/casualvae/module_aclf_partial_multimodal.py
data/structured_full/*.csv
configs/aclf_full_multimodal_config.json
artifacts/multiseed/fixed_split
```

Run child training with repository root as its working directory.

- [ ] **Step 5: Update analysis defaults**

Use:

```text
artifacts/multiseed/variable_split
artifacts/multiseed/fixed_split
artifacts/analysis/multiseed
```

- [ ] **Step 6: Run focused tests and verify GREEN**

Run:

```powershell
conda run -n casualvae_01 python -m unittest tests.test_project_layout tests.test_run_multiseed tests.test_seed_separation tests.test_stability_statistics -v
```

Expected: all focused tests pass.

---

### Task 5: Update user documentation and remove disposable files

**Files:**
- Modify: `readme.md`
- Delete: `~$chnical_Proposal_V4.docx`
- Delete: `__pycache__/`
- Delete: `.pytest_cache/`
- Delete: `tests/__pycache__/` if regenerated

**Interfaces:**
- Consumes: canonical command and directory paths.
- Produces: one authoritative project map and runnable commands.

- [ ] **Step 1: Replace the README tree and path descriptions**

Document `src`, `scripts`, `configs`, `data`, `artifacts`, `docs`, and `tests` with their
responsibilities.

- [ ] **Step 2: Update commands**

Use commands rooted at the repository, including:

```powershell
conda run -n casualvae_01 python scripts/preprocessing/prepare_full_augmented_structured.py
conda run -n casualvae_01 python src/casualvae/module_aclf_partial_multimodal.py --config_json configs/aclf_full_multimodal_config.json ...
conda run -n casualvae_01 python scripts/training/run_multiseed.py --split-seed 42 --model-seeds 42 123 456 789 2026 --device cpu
conda run -n casualvae_01 python scripts/analysis/analyze_multiseed.py
```

- [ ] **Step 3: Delete only disposable files**

Validate each cache/lock-file path is inside the repository, then remove it. Do not
delete any data or artifact.

- [ ] **Step 4: Scan for stale paths**

Run:

```powershell
rg -n "working space/casualmodule|outputs[/\\]|python (module_aclf_partial_multimodal|run_multiseed|analyze_multiseed)\.py" -g "*.py" -g "*.md" .
```

Expected: no executable or documented stale path remains.

---

### Task 6: Verify behavior and artifacts

**Files:**
- Verify: all relocated project files and generated outputs

**Interfaces:**
- Consumes: completed migration.
- Produces: evidence that layout changed without behavior or data loss.

- [ ] **Step 1: Compile all Python files**

Run:

```powershell
conda run -n casualvae_01 python -m compileall -q src scripts tests
```

Expected: exit code 0.

- [ ] **Step 2: Run the complete regression suite**

Run:

```powershell
conda run -n casualvae_01 python -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 3: Verify fixed-split result completeness and patient IDs**

For model seeds `42, 123, 456, 789, 2026`, verify required files in
`artifacts/multiseed/fixed_split/model_seed_<seed>` and exact equality of all six patient
ID arrays in `data_split_summary.json`.

- [ ] **Step 4: Re-run stability analysis from the new entry point**

Run:

```powershell
conda run -n casualvae_01 python scripts/analysis/analyze_multiseed.py
```

Expected: both experiments are analyzed and
`artifacts/analysis/multiseed/stability_comparison.csv` is written.

- [ ] **Step 5: Compare pre- and post-migration metrics**

Expected fixed-split metrics:

```text
mean_top50_jaccard       0.007419676507849653
mean_top100_jaccard      0.010806437503576892
mean_spearman            0.034627936926669226
strict_consensus_count   0
```

- [ ] **Step 6: Inspect final Git and filesystem state**

Confirm the root contains only project metadata and top-level category directories,
no generated dataset is staged, and unrelated user changes remain present.
