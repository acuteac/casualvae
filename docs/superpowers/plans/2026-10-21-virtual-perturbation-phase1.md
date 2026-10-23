# Virtual perturbation phase 1 implementation plan

**Goal:** Prepare auditable candidates, patient cohorts and input-SD dose plans without inference.

**Architecture:** A standalone analysis CLI consumes the existing 4.2/4.3 schemas, model metadata and structured inputs. Reuse existing edge ranking, model registry and summary helpers. Validate before publishing manifests; write a diagnostic preflight report on failure. Never infer missing historical EID splits from counts or directory names.

**Tech stack:** Python, pandas, numpy, existing analysis helpers; synthetic unittest/pytest fixtures.

- [x] Add `tests/test_prepare_virtual_perturbation.py`: synthetic full edge tables for both modalities, four models and shuffled EID tables. Assert lexicographic tie breaks, distinct support/frequency, controls, missingness, split auditing, training quantiles, dose rows and failure diagnostics. Run before implementation and confirm missing functionality fails.
- [x] Add `scripts/analysis/prepare_virtual_perturbation.py`: stable output schemas, validated loading, candidate selection, model-specific cohort and dose preparation, hashes/config/report, dry-run and protected output paths. No torch/model inference imports.
- [x] Run focused tests; correct discrepancies against the written selection and cohort contracts.
- [x] Add `docs/virtual_perturbation_phase1.md` with CLI, exact field definitions, rules, and phase 2 prerequisites; link from `readme.md`.
- [x] Run real preflight and preparation with existing local inputs, report missing historical split EIDs without fabrication. Run full pytest if available, inspect diff and mark verification outcomes.

Selection contract: within each modality, sort Primary Top20 Outcome frequency descending, Top50 frequency descending, mean Total rank over all Outcomes ascending, mean four-model Top20 support on Primary Top20 edges descending (fallback: best Primary edge(s)), mean Total sign agreement over all Outcomes descending, original feature ascending. Controls use seeded sampling of the lowest-effect quintile ordered by worst best-rank percentile across Outcomes, then mean rank percentile; exclude candidates.

Cohort contract: reproduce existing finite-float32 availability in config feature order after EID joins. Audit every patient/candidate. Main cohort is paired validation with both modalities available and candidate observed. Strictly untrained excludes the union of all three training stages. Single-modality validation is labeled exploratory and never enabled for the joint executor. Quantiles use observed values in the corresponding modality training union paired training only. Preserve zero SD even beyond quantiles; flag its support status while allowing eligible observed baselines.

Review: an independent read-only review found incomplete architecture/container/noise validation. Reproduced with a failing synthetic regression and added explicit dimension, mapping, hidden-layer and finite sigma checks. Historical inactive outcome hidden-layer metadata is warned and preserved; active configuration mismatches remain fatal.

Verification: full pytest passed 78 tests (18 phase-1 tests). Real preparation is correctly blocked by missing six EID split lists and seed metadata in both historical variable-split models. Only preflight_report.json was published; it contains real candidate previews and valid Primary/Sensitivity cohort diagnostics. All 29 recorded input hashes remained unchanged. Formal four-model manifests await authentic historical split provenance.
