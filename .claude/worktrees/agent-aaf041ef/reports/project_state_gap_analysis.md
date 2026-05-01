# Project State Gap Analysis

**Date**: 2026-04-24 (revised)
**Scope**: Full comparison of current repository state vs. revised experiment structure

---

## SECTION 1 -- Executive Summary

### What is in good shape

The Stage 2 meta-learning framework itself is in good shape. The project is not missing core meta-learning machinery. All five adaptation methods (zero-shot, fine-tune, ANIL, FOMAML, MAML) are implemented, tested, and have produced verified results. The shared runner, grid-based experiment config, contract-aware episode sampling, gold-label evaluation with repeated episodes, collapse detection, and the 60+ function plotting library are all functional and well-integrated.

- **Stage 2 meta-learning core** is fully implemented and well-tested: MAML, FOMAML, ANIL, zero-shot, and fine-tune all have working code, a shared runner, and a grid-based experiment config system.
- **67 completed Stage 2 experiments** exist in the registry, covering all 5 methods on Logistics with the A_weak_only initialisation. The LR sweep (Group B) and richer department check (Group E) have been run and saved.
- **Stage 1 code** (`stage1.py`, `baselines.py`, `mlp.py`) supports all 4 model families (Mean, ElasticNet, XGBoost, MLP) across multiple supervision conditions, with modular training, evaluation, and saving.
- **Plotting/dashboard** is extensive: 60+ plotting functions in `plotting.py`, including Stage 2-specific visualisations for LR sweeps, collapse heatmaps, k-shot curves, and initialisation ablation comparisons.
- **Episode sampling** is contract-aware and deterministically seeded.
- **Evaluation** is gold-label based, with repeated episodes (N=20), mean/std aggregation, collapse detection, and ranking metrics.

### Where the gaps are

The main missing pieces are not in the meta-learning machinery but in the **fair comparison layer** -- the complete comparable experiment structure needed for the thesis. Specifically, what is missing is:

1. **All initialisation sources**: only A_weak_only has been used in Stage 2. B_gold_only, C_hybrid, D_hybrid_unweighted, and random are not yet run.
2. **Full initialisation x method matrix**: the 5x5 grid (5 init sources x 5 adaptation methods) is filled only for the A_weak_only row.
3. **Repeated contract-sample robustness**: while repeated episodes (N=20) are technically supported, the final comparable matrix across all init sources and departments has not been produced with consistent repeated-episode reporting.
4. **Department-as-target evaluation**: only partial department coverage exists (Logistics fully, a few experiments on Devices and Needles and Packaging Material).
5. **K-shot / support-size comparison**: the preset exists but no results have been generated.
6. **Weighting ablation**: D_hybrid_unweighted does not exist at all, so the weighted vs. unweighted hybrid comparison cannot be made.

### Additional gaps

7. **D_hybrid_unweighted does not exist at all**: not in Stage 1 artifacts, not in config, not mentioned anywhere in code.
8. **B_gold_only is not wired into Stage 2**: Stage 1 artifacts exist, but the Stage 2 config does not reference it.
9. **Stage 1 CLI runner is a stub**: `run_stage1.py` has only commented-out pseudocode.
10. **Stage 1 does not save gold-label evaluation** as separate artifacts.
11. **MLP architecture mismatch**: Stage 1 MLPs use `hidden_dim_1=256, hidden_dim_2=64`, but the Stage 2 config specifies `hidden_dim_1=256, hidden_dim_2=128`.

### Immediate next priorities

1. Verify the MLP dim mismatch between Stage 1 artifacts and Stage 2 config.
2. Run Group C (initialisation ablation: A_weak_only vs C_hybrid) -- the preset is already defined.
3. Run Group D (k-shot sweep) -- the preset is already defined.
4. Add B_gold_only to the Stage 2 experiment grid.
5. Save Stage 1 gold evaluation artifacts.
6. Train D_hybrid_unweighted in Stage 1 and save artifacts.
7. Fix random-init preprocessor support and run random-init experiments.
8. Complete the full init x method matrix across all departments with repeated-episode summaries.

---

## SECTION 2 -- Current Pipeline Overview

### Data flow

```
Data/raw/ --> notebooks 01-07 (join) --> Data/processed/contract_with_features_labeled_with_gold.csv
  --> notebook 15 (Snorkel labeling) --> weak labels (renegotiation_prob)
  --> notebook 15.1 (gold labels) --> gold_y column
  --> notebook 16 (Stage 1 baselines) --> models/stage_1/{A_weak_only,B_gold_only,C_hybrid}/
  --> notebook 17 / experiments/run_stage2.py --> models/stage_2/experiments/{experiment_id}/
  --> notebook 17.1 (dashboard) --> reports/
```

### Source modules

| File | Role |
|---|---|
| [config.py](file:///Users/Thomas/Desktop/Master%20Thesis/src/master_thesis/config.py) | Central paths and seed |
| [data_utils.py](file:///Users/Thomas/Desktop/Master%20Thesis/src/master_thesis/data_utils.py) | Data loading and splitting |
| [snorkel_lfs.py](file:///Users/Thomas/Desktop/Master%20Thesis/src/master_thesis/snorkel_lfs.py) | Labeling functions for weak supervision |
| [baselines.py](file:///Users/Thomas/Desktop/Master%20Thesis/src/master_thesis/baselines.py) | Mean, ElasticNet, XGBoost baseline models |
| [mlp.py](file:///Users/Thomas/Desktop/Master%20Thesis/src/master_thesis/mlp.py) | TabularMLP architecture + training loop |
| [stage1.py](file:///Users/Thomas/Desktop/Master%20Thesis/src/master_thesis/stage1.py) | Stage 1 orchestration (multi-condition) |
| [episode_sampler.py](file:///Users/Thomas/Desktop/Master%20Thesis/src/master_thesis/episode_sampler.py) | Contract-aware episodic task construction |
| [maml.py](file:///Users/Thomas/Desktop/Master%20Thesis/src/master_thesis/maml.py) | Full MAML implementation |
| [FOMAML.py](file:///Users/Thomas/Desktop/Master%20Thesis/src/master_thesis/FOMAML.py) | First-order MAML implementation |
| [anil.py](file:///Users/Thomas/Desktop/Master%20Thesis/src/master_thesis/anil.py) | ANIL implementation |
| [meta_common.py](file:///Users/Thomas/Desktop/Master%20Thesis/src/master_thesis/meta_common.py) | Shared meta-learning utilities (zero-shot, collapse, stats) |
| [stage2.py](file:///Users/Thomas/Desktop/Master%20Thesis/src/master_thesis/stage2.py) | Stage 2 orchestration, config resolution, experiment grid |
| [metrics.py](file:///Users/Thomas/Desktop/Master%20Thesis/src/master_thesis/metrics.py) | All evaluation metrics (gold, weak, ranking, calibration) |
| [plotting.py](file:///Users/Thomas/Desktop/Master%20Thesis/src/master_thesis/plotting.py) | 60+ plotting functions for thesis figures |
| [explainability.py](file:///Users/Thomas/Desktop/Master%20Thesis/src/master_thesis/explainability.py) | SHAP-based explainability |

### Saved artifacts

**Stage 1** (3 conditions trained):
- `models/stage_1/A_weak_only/` -- MLP weights, preprocessor, history, tabular results, XGBoost importance
- `models/stage_1/B_gold_only/` -- same structure
- `models/stage_1/C_hybrid/` -- same structure (weighted hybrid)
- Legacy files at root: `models/stage_1/mlp_pretrained.pt` (appears to be the A_weak_only copy)

**Stage 2** (67 experiments):
- `models/stage_2/experiments/experiment_summary.csv` -- master registry
- 67 experiment directories, each containing: `metrics.csv`, `predictions.csv`, `history.csv`, `prediction_stats.csv`, `resolved_config.yaml`, `stage2_result_summary.json`, and model weights

---

## SECTION 3 -- Stage 1 Status Table

| Stage 1 Component | Expected | Current Status | Evidence | Gap / Issue |
|---|---|---|---|---|
| **A_weak_only** | Trained on weak labels | Implemented, trained, saved | `models/stage_1/A_weak_only/` (MLP + preprocessor + baselines) | None -- complete |
| **B_gold_only** | Trained on gold labels only | Implemented, trained, saved | `models/stage_1/B_gold_only/` (MLP + preprocessor + baselines) | Not referenced in Stage 2 config |
| **C_hybrid_weighted** | Weighted hybrid (gold_weight=5x) | Implemented, trained, saved | `models/stage_1/C_hybrid/` -- note: folder named `C_hybrid`, not `C_hybrid_weighted` | In Stage 2 config as `C_hybrid`. Naming ambiguity with unweighted variant |
| **D_hybrid_unweighted** | Equal-weight hybrid | **Not implemented** | No folder, no code reference, no config entry | Entirely missing. `stage1_config.yaml` has weights `(1.0, 5.0)` but no unweighted option |
| **Mean Predictor** | Baseline | Implemented, runnable | `baselines.py:fit_mean_predictor()`, results in `tabular_results.csv` | Complete |
| **ElasticNet** | Baseline | Implemented, runnable | `baselines.py:fit_elastic_net()`, results in `tabular_results.csv` | Complete |
| **XGBoost** | Baseline | Implemented, runnable | `baselines.py:fit_xgboost_regressor()`, results in `tabular_results.csv` | Complete |
| **MLP** | Baseline + Stage 2 backbone | Implemented, runnable | `mlp.py:TabularMLP`, `mlp.py:fit_mlp_model()` | **Dim mismatch**: Stage 1 saves `hidden_dim_2=64` but Stage 2 config expects `hidden_dim_2=128` |
| **Stage 1 CLI runner** | Automated training script | **Stub only** | `experiments/run_stage1.py` -- all logic is commented out | Not usable as-is; all training runs through notebook 16 |
| **Stage 1 gold evaluation** | Gold-label evaluation table per condition | **Not saved** | `stage1.py:evaluate_stage1_on_gold_labels()` exists but results are only in notebook memory, not persisted as artifacts | No `gold_results.csv` in any `models/stage_1/` directory |

> [!WARNING]
> **MLP architecture mismatch**: Stage 1 A_weak_only was trained with `hidden_dim_2=64` (per `stage1_config.yaml`), but Stage 2 `mlp_config` specifies `hidden_dim_2=128`. When Stage 2 creates a `TabularMLP(hidden_dim_2=128)` and loads the `A_weak_only` weights (which are for `hidden_dim_2=64`), `load_state_dict()` will either fail or silently produce wrong results due to shape mismatches. **This needs verification** -- check whether Stage 2 experiments use the config MLP dims or whether the successful runs indicate the dims actually match.

---

## SECTION 4 -- Stage 2 Status Table

| Stage 2 Component | Expected | Current Status | Evidence | Gap / Issue |
|---|---|---|---|---|
| **Zero-shot** | No-adapt baseline | Implemented, run | 5 experiments in registry | Complete |
| **Fine-tune** | Per-episode gradient-based fine-tuning | Implemented, run | 4 experiments in registry | Complete |
| **ANIL** | Head-only adaptation | Implemented, run | 6 experiments in registry | Complete |
| **FOMAML** | First-order MAML | Implemented, run | 26 experiments in registry (incl. LR sweep) | Complete |
| **MAML** | Full second-order MAML | Implemented, run | 26 experiments in registry (incl. LR sweep) | Complete |
| **Repeated episodes** | N=20 support/query repeats | Implemented | `n_repeats: 20` in config, metrics confirm 20 rows per experiment | Complete |
| **Per-department analysis** | Multi-department evaluation | Partially run | 4 experiments on Devices and Needles, 4 on Packaging Material | Only A_weak_only init, only some methods |
| **K-shot sweep** | k=1..5 support size sweep | **Config exists, not run** | `k_shot_sweep_logistics` preset defined; no k=1,3,4 experiments in registry | Preset ready, no results |
| **Initialisation ablation** | A_weak vs C_hybrid vs random | **Config exists, not run** | `initialization_ablation` preset defined; all experiments are A_weak_only | Preset ready, no C_hybrid or random results |
| **LR sweep** | Inner LR sensitivity | Implemented, run | Groups B covers lr=0.0005 to 0.01 x steps=1 to 5 for FOMAML/MAML | Complete for A_weak_only |
| **Collapse detection** | Prediction stats + collapse rate | Implemented | `prediction_stats.csv` saved per experiment, `collapse_rate` in summary | Complete |

---

## SECTION 5 -- Initialisation x Method Matrix

**Legend**: Run = completed experiments exist; Config = preset/config supports it but not yet run; Gap = no support in config or code.

| Init Source | zero-shot | fine-tune | ANIL | FOMAML | MAML |
|---|---|---|---|---|---|
| **A_weak_only** | Run (5) | Run (4) | Run (6) | Run (26) | Run (26) |
| **B_gold_only** | Gap | Gap | Gap | Gap | Gap |
| **C_hybrid** (weighted) | Config | Config | Config | Config | Config |
| **D_hybrid_unweighted** | Gap | Gap | Gap | Gap | Gap |
| **Random** | Config | Config | Config | Config | Config |

### Assessment

- **A_weak_only**: fully covered. 67 experiments across all methods, multiple LR values, multiple step counts, multiple departments.
- **B_gold_only**: Stage 1 artifacts exist (`models/stage_1/B_gold_only/mlp_pretrained.pt`), but it is not mentioned in `stage2_config.yaml` and has never been used as an init source for Stage 2.
- **C_hybrid** (weighted): Stage 1 artifacts exist, referenced in `stage2_config.yaml` (init_names lists it), but **zero experiments** have been run with it. The `initialization_ablation` preset is ready.
- **D_hybrid_unweighted**: does not exist anywhere -- no Stage 1 artifacts, no config, no code.
- **Random**: referenced in `stage2_config.yaml` `experiment_grid.init_names` and in `zero_shot_only` preset. Code path exists in `stage2.py` (line 218: `if init_type == "random"`). But the preprocessor problem (line 488-490: raises error when preprocessor is None) means random init is **partially supported** -- it works only if a preprocessor is provided externally. **No experiments have been run.**

### Final thesis comparison target

The final fair thesis comparison should fill the full matrix below, where rows are initialisation sources and columns are adaptation methods:

| | zero-shot | fine-tune | ANIL | FOMAML | MAML |
|---|---|---|---|---|---|
| **A_weak_only** | | | | | |
| **B_gold_only** | | | | | |
| **C_hybrid** (weighted) | | | | | |
| **D_hybrid_unweighted** | | | | | |
| **Random** | | | | | |

Each cell should be evaluated:
- per department (each department as a target)
- with multiple contract-sample draws per department (repeated episodes), reported as mean +/- std
- with gold-label evaluation as the primary basis
- using AUROC, Log-Loss, NDCG@10, and ECE as the main metrics

---

## SECTION 6 -- Missing Items and Inconsistencies

### Critical gaps

1. **D_hybrid_unweighted condition does not exist at all.**
   - No training code, no Stage 1 artifacts, no config entry.
   - Needs: create the condition in the Stage 1 pipeline (set `sample_weight=None` for both weak and gold labels combined), train all 4 baselines + MLP, save artifacts.

2. **B_gold_only not in Stage 2 config.**
   - Stage 1 artifacts exist and look correct.
   - Needs: add `"B_gold_only"` to `init_names` in relevant presets (at minimum in `initialization_ablation`).

3. **MLP hidden_dim_2 mismatch (64 vs 128).**
   - `stage1_config.yaml`: `hidden_dim_2: 64`
   - `stage2_config.yaml`: `hidden_dim_2: 128`
   - If the successful experiments loaded A_weak_only weights (which are dim_2=64) into a dim_2=128 model, `load_state_dict` would raise a shape mismatch error -- so either (a) Stage 2 is actually using dim_2=64 despite the config saying 128 (because the notebook overrides it), or (b) there is a bug. **Requires verification.**

4. **Random init preprocessor problem.**
   - `stage2.py` line 488-490 raises `ValueError` when `preprocessor is None`.
   - For random init, the Stage 1 preprocessor path is set to None.
   - Needs: either (a) supply a shared preprocessor for random init, or (b) fall back to a dynamically fitted preprocessor when `init_name == "random"`.

5. **Confusion matrices not integrated as standard supplementary output.**
   - No experiment currently produces a confusion matrix.
   - Confusion matrices should be generated for selected best models and key comparisons, evaluated on gold labels, reported per department.
   - They are supplementary diagnostics and should not replace AUROC, Log-Loss, NDCG@10, and ECE as the main metrics, but they provide interpretive value (e.g. showing whether errors are false positives vs. false negatives).

6. **Repeated contract-sample robustness not explicit in final design.**
   - Repeated episodes (N=20) are technically supported and used in existing experiments.
   - However, the final comparable matrix across all init sources and departments has not been produced with consistent repeated-episode reporting.
   - Each department should be evaluated with multiple different support/query contract samples, and results should be reported as mean +/- std or equivalent repeated-episode summaries. This is necessary so that results are not driven by one favorable or unlucky split.
   - This requirement needs to be made explicit in the final experiment design and reporting.

7. **Full department-as-target evaluation is still incomplete.**
   - Logistics is fully covered for A_weak_only. A few experiments exist for Devices and Needles and Packaging Material.
   - The final comparable matrix across all init sources and all departments has not been produced.

### Moderate gaps

8. **Group C (initialisation ablation) not yet run.**
   - Preset `initialization_ablation` is defined and ready.
   - Needs: `bash experiments/run_experiment_groups.sh C`

9. **Group D (k-shot sweep) not yet run.**
   - Preset `k_shot_sweep_logistics` is defined and ready.
   - Needs: `bash experiments/run_experiment_groups.sh D`

10. **Stage 1 CLI runner is a stub.**
    - `experiments/run_stage1.py` contains only commented-out pseudocode.
    - All Stage 1 training currently happens in notebook 16.
    - Not blocking, but prevents reproducible CLI-based Stage 1 runs.

11. **Stage 1 gold evaluation not saved as artifacts.**
    - `evaluate_stage1_on_gold_labels()` exists but results are transient (in notebook memory only).
    - Saved `tabular_results.csv` only contains weak-distillation metrics (RMSE, MAE, R2), not gold metrics (AUROC, F1, etc.).
    - Needs: save `gold_results.csv` alongside `tabular_results.csv` in each `models/stage_1/` directory.

### Minor / cosmetic issues

12. **Naming inconsistency: `C_hybrid` vs `C_hybrid_weighted`.**
    - The directory and config use `C_hybrid`, but the revised structure calls it `C_hybrid_weighted`.
    - Recommendation: keep `C_hybrid` everywhere (less risky than renaming) and document that it refers to the weighted variant.

13. **Duplicate `support_config` in `stage2_config.yaml`.**
    - Lines 22-24 and 47-49 both define `support_config`. The second one overrides the first. Not harmful, but confusing.

14. **Code duplication in meta-learner files.**
    - `_prepare_tensors()` and `_safe_buffers()` are identical in `maml.py`, `FOMAML.py`, and partially in `meta_common.py`. Not a correctness issue, but a maintenance risk.

15. **Legacy models at `models/stage_1/` root.**
    - `mlp_pretrained.pt`, `mlp_preprocessor.joblib`, `mlp_training_history.csv` at the root level appear to be pre-refactor copies. Should be either removed or documented.

---

## SECTION 7 -- What Is Already Enough for Supervisor Discussion

### Presentable now

1. **Group A findings**: All 5 methods (zero-shot, finetune, ANIL, FOMAML, MAML) compared at default settings on Logistics with A_weak_only init. Clear baseline comparison is possible.

2. **Group B stabilisation findings**: Full FOMAML/MAML LR sweep (5 LR values x 4 step counts = 40 experiments). You can show:
   - LR sensitivity heatmaps
   - Collapse rate analysis (which LR/steps combinations cause collapse)
   - Optimal stable configurations (FOMAML: steps=5, lr=0.001; MAML: steps=3, lr=0.001)

3. **Group E richer department check**: 8 experiments on Devices and Needles and Packaging Material with stabilised settings. Shows whether meta-learning generalises beyond Logistics.

4. **Current evidence for method behaviour**: With 67 experiments, you have enough data to characterise:
   - ANIL tends to be more stable than FOMAML/MAML
   - FOMAML and MAML are sensitive to inner_lr
   - Collapse is the primary failure mode at high learning rates

5. **Numerical stability measures**: BatchNorm clamping, gradient clipping, collapse detection are all implemented and documented -- shows engineering rigour.

Groups A, B, and E already provide enough evidence to discuss current method behaviour with the supervisor and to demonstrate that the meta-learning pipeline is functional and producing interpretable results. However, they do not yet complete the final fair thesis comparison matrix across all initialisation sources, departments, and repeated-episode conditions.

### Not presentable yet

- Initialisation ablation (C_hybrid vs A_weak_only) -- no C_hybrid runs exist
- K-shot analysis -- no k=1,3,4 results exist
- B_gold_only or D_hybrid_unweighted comparisons -- no results exist
- Stage 1 gold-label cross-condition comparison table -- not saved as persistent artifacts
- Full init x method matrix with consistent repeated-episode reporting across departments

---

## SECTION 8 -- Recommended Next Steps

### Priority 1 -- Must do for thesis comparability

| # | Task | Effort | Dependency |
|---|---|---|---|
| 1.1 | **Verify MLP dim mismatch**: check whether `models/stage_1/A_weak_only/mlp_pretrained.pt` actually has `hidden_dim_2=64` or `128`. If 64, update `stage2_config.yaml` to match. | 5 min | None |
| 1.2 | **Run Group C** (`initialization_ablation`): ANIL/FOMAML/MAML x {A_weak_only, C_hybrid} x steps=3 x lr=0.005 | 30-60 min | C_hybrid Stage 1 artifacts exist |
| 1.3 | **Run Group D** (`k_shot_sweep_logistics`): 4 methods x k={1,2,3,4} x steps=3 | 1-2 hr | None |
| 1.4 | **Add B_gold_only** to `stage2_config.yaml` init_names and run at least a baseline comparison (zero_shot + FOMAML/MAML at best-known LR) | 1-2 hr | B_gold_only Stage 1 artifacts exist |
| 1.5 | **Save Stage 1 gold-label evaluation** as `gold_results.csv` in each `models/stage_1/` directory | 30 min | Re-run notebook 16 evaluation cells |

### Priority 2 -- Important for thesis completeness

| # | Task | Effort | Dependency |
|---|---|---|---|
| 2.1 | **Train D_hybrid_unweighted** in Stage 1: combine weak + gold labels with equal weight (sample_weight=1.0 for both), save all 4 models + MLP | 1-2 hr | Notebook 16 modification |
| 2.2 | **Fix random init preprocessor path**: supply a shared preprocessor (e.g. from A_weak_only) when `init_name=="random"`, or fit one dynamically | 30 min | Code change in stage2.py |
| 2.3 | **Run random init experiments**: at least zero_shot + FOMAML/MAML at stable LR | 1-2 hr | 2.2 |
| 2.4 | **Complete the init x method matrix**: after 2.1-2.3, run the remaining cells to fill the 5x5 matrix | 4-8 hr | 1.2, 1.4, 2.1, 2.3 |

Once the core init x method matrix is in place, confusion matrices should be generated for selected best comparisons per department as supplementary diagnostics.

### Priority 3 -- Nice to have / technical cleanup

| # | Task | Effort | Dependency |
|---|---|---|---|
| 3.1 | Implement `run_stage1.py` as a working CLI script (not just a stub) | 1-2 hr | None |
| 3.2 | Deduplicate `_prepare_tensors` / `_safe_buffers` across maml.py, FOMAML.py, meta_common.py | 30 min | None |
| 3.3 | Remove legacy root-level files in `models/stage_1/` | 5 min | Verify they are duplicates |
| 3.4 | Fix duplicate `support_config` in `stage2_config.yaml` | 5 min | None |
| 3.5 | Add optional L1 regularisation parameter to meta-training configs | 30 min | None |
| 3.6 | Add partial layer freezing option to MAML/FOMAML (freeze feature extractor, adapt head + last block) | 1-2 hr | None |

---

## SECTION 9 -- Optional Implementation Plan

### Phase 1: Quick wins (1 day)

1. Verify MLP dim mismatch (task 1.1)
2. Fix duplicate `support_config` in config (task 3.4)
3. Run Group C: `bash experiments/run_experiment_groups.sh C` (task 1.2)
4. Run Group D: `bash experiments/run_experiment_groups.sh D` (task 1.3)
5. Save Stage 1 gold evaluation artifacts (task 1.5)

### Phase 2: Init ablation expansion (1-2 days)

6. Add `B_gold_only` to `stage2_config.yaml` init_names in relevant presets (task 1.4)
7. Create `D_hybrid_unweighted` condition in notebook 16 and save Stage 1 artifacts (task 2.1)
8. Fix random init preprocessor issue in `stage2.py` (task 2.2)
9. Create a new preset `full_init_ablation` covering all 5 init sources x {zero_shot, FOMAML, MAML} x steps=3 x lr=0.001

### Phase 3: Fill the matrix and ensure comparability (2-3 days)

10. Run the full_init_ablation preset across all departments
11. Ensure repeated contract-sample robustness: each department evaluated with multiple support/query draws, results reported as mean +/- std
12. Ensure department-as-target comparability: every department appears as a target in the final matrix
13. Update the analysis dashboard (notebook 17.1) to display the 5x5 matrix
14. Generate final thesis tables and figures

### Phase 4: Supplementary outputs and cleanup (1 day)

15. Generate confusion matrices for selected best models per department (supplementary, gold-label-based)
16. Implement `run_stage1.py` properly (task 3.1)
17. Deduplicate shared functions (task 3.2)
18. Clean up legacy artifacts (task 3.3)
19. Final documentation pass

---

## SECTION 10 -- What the final comparable setup should capture

The final thesis comparison should capture all of the following dimensions:

1. **Each department as a target department.** Results should not be limited to Logistics; every department with sufficient gold-labelled contracts should appear as a target.

2. **Multiple support/query contract draws per department.** Each department-target evaluation should use multiple different support/query contract samples (repeated episodes). Results should be reported as mean +/- std or equivalent summary statistics so that conclusions are not driven by one favorable or unlucky split.

3. **Repeated-episode summary statistics.** Per-cell values in the init x method matrix should be aggregated over repeated episodes, not single-run point estimates.

4. **The full initialisation x adaptation-method matrix.** All five rows (A_weak_only, B_gold_only, C_hybrid, D_hybrid_unweighted, Random) crossed with all five columns (zero-shot, fine-tune, ANIL, FOMAML, MAML).

5. **K-shot sensitivity.** At least a sweep over support sizes (e.g. k=1..5) for representative init sources and methods.

6. **Gold-label-based comparison as the main evaluation basis.** AUROC, Log-Loss, NDCG@10, and ECE evaluated on gold labels are the primary metrics.

7. **Confusion matrices as supplementary interpretation only.** Reported for selected best models and key comparisons, per department, on gold labels. They do not replace the main metrics but provide qualitative insight into error patterns (e.g. false positive vs. false negative rates).

This applies to both categories of initialisation:

- **Models starting from pretrained initialisations:**
  - A_weak_only (trained on weak labels only)
  - B_gold_only (trained on gold labels only)
  - C_hybrid (weighted combination of weak and gold labels)
  - D_hybrid_unweighted (equal-weight combination of weak and gold labels)

- **Models starting from a random / untrained initialisation:**
  - Random (no pretraining)

The final comparable setup should not only compare adaptation methods within pretrained models, but also compare whether the same adaptation methods behave differently when starting from no pretraining at all. This is the key contrast that makes the initialisation ablation scientifically meaningful: it isolates the effect of pretraining quality on few-shot adaptation.
