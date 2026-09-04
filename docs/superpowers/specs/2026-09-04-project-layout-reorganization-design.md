# Project Layout Reorganization Design

## Objective

Reorganize the CasualVAE workspace by responsibility while preserving every dataset,
training artifact, analysis result, and executable behavior. The migration may update
paths, imports, package markers, documentation, and ignore rules, but must not change
model architecture, losses, zero-mode behavior, SCM behavior, training algorithms, or
statistical formulas.

## Target Layout

```text
casualvae/
├─ src/
│  └─ casualvae/
│     ├─ __init__.py
│     └─ module_aclf_partial_multimodal.py
├─ scripts/
│  ├─ preprocessing/
│  │  ├─ data_augmentation.py
│  │  ├─ prepare_augmented_structured.py
│  │  ├─ prepare_full_augmented_structured.py
│  │  └─ generate_full_config.py
│  ├─ training/
│  │  └─ run_multiseed.py
│  ├─ analysis/
│  │  └─ analyze_multiseed.py
│  └─ utilities/
│     └─ extract.py
├─ configs/
│  └─ aclf_full_multimodal_config.json
├─ data/
│  ├─ augmented/
│  └─ structured_full/
├─ artifacts/
│  ├─ runs/
│  │  ├─ run_fixed_full/
│  │  ├─ run_verify_seed42_20260904_v2/
│  │  └─ run_zero_seed42_20260904/
│  ├─ multiseed/
│  │  ├─ variable_split/
│  │  └─ fixed_split/
│  └─ analysis/
│     └─ multiseed/
├─ docs/
│  ├─ Technical_Proposal_V4.docx
│  ├─ assets/
│  │  └─ 模型架构.png
│  └─ superpowers/
│     └─ specs/
├─ tests/
├─ readme.md
└─ requirements.txt
```

## File Mapping

| Current path | Destination |
|---|---|
| `module_aclf_partial_multimodal.py` | `src/casualvae/module_aclf_partial_multimodal.py` |
| `data_augmentation.py` | `scripts/preprocessing/data_augmentation.py` |
| `prepare_augmented_structured.py` | `scripts/preprocessing/prepare_augmented_structured.py` |
| `prepare_full_augmented_structured.py` | `scripts/preprocessing/prepare_full_augmented_structured.py` |
| `generate_full_config.py` | `scripts/preprocessing/generate_full_config.py` |
| `run_multiseed.py` | `scripts/training/run_multiseed.py` |
| `analyze_multiseed.py` | `scripts/analysis/analyze_multiseed.py` |
| `extract.py` | `scripts/utilities/extract.py` |
| `aclf_full_multimodal_config.json` | `configs/aclf_full_multimodal_config.json` |
| `outputs/augmented` | `data/augmented` |
| `outputs/structured_full` | `data/structured_full` |
| `outputs/run_*` | `artifacts/runs/run_*` |
| `outputs/multiseed` | `artifacts/multiseed/variable_split` |
| `outputs/multiseed_fixed_split` | `artifacts/multiseed/fixed_split` |
| `outputs/multiseed_analysis` | `artifacts/analysis/multiseed` |
| `Technical_Proposal_V4.docx` | `docs/Technical_Proposal_V4.docx` |
| `模型架构.png` | `docs/assets/模型架构.png` |

The Word lock file `~$chnical_Proposal_V4.docx`, Python bytecode caches, and pytest
cache are disposable generated files and will be removed rather than migrated.

## Path and Import Migration

All executable paths will be resolved from the repository root derived from each
script's `__file__`, not from the caller's current working directory and not from an
absolute machine-specific path. Defaults will become:

- structured inputs: `data/structured_full/`
- augmented inputs: `data/augmented/`
- configuration: `configs/aclf_full_multimodal_config.json`
- fixed-split runs: `artifacts/multiseed/fixed_split/`
- variable-split runs: `artifacts/multiseed/variable_split/`
- multi-seed analysis: `artifacts/analysis/multiseed/`

Tests will import the model module from `src.casualvae` and load script modules from
their new packages. Package marker files will contain no runtime logic.

## Compatibility Policy

The old file locations will not be retained as duplicate files or symbolic links.
Commands documented in `readme.md` will be updated to the canonical new paths. Existing
CSV, JSON, model weights, histories, and analysis results will retain their names and
contents; only their parent directories change.

## Migration Safety

Before every move, the absolute source and destination will be checked to ensure both
are inside the repository. Existing destinations will not be overwritten. Moves will
remain on the same drive and will be performed in small responsibility-based groups.
Unrelated working-tree changes will be preserved.

## Verification

After migration:

1. Run Python syntax compilation using the `casualvae_01` environment.
2. Run the complete unittest suite using the `casualvae_01` environment.
3. Confirm all five fixed-split run directories contain their required model, history,
   summary, and causal graph files.
4. Confirm fixed-split patient IDs remain identical across model seeds.
5. Run the analysis CLI against the relocated variable- and fixed-split results.
6. Confirm the regenerated comparison metrics match the pre-migration values.
7. Search for stale project-specific absolute paths and old `outputs/` references.

## Non-goals

- No model module decomposition or renaming of public classes/functions.
- No changes to architecture, losses, zero-mode, SCM, optimizers, seed semantics, or
  statistical definitions.
- No deletion of datasets, trained models, or analysis artifacts.
- No dependency upgrades or environment changes.
