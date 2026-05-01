# Project Handoff Summary
## Contract Renegotiation Prediction — Stage 1 → Stage 2 Meta-Learning Pipeline

*Generated: 2026-04-26 | Codebase snapshot: master branch*

---

## 1. Repository Overview

**Root:** `Master Thesis/`

```
Master Thesis/
├── src/master_thesis/          # All production Python modules
│   ├── config.py               # Central path constants + SEED=42
│   ├── stage1.py               # Stage 1 orchestration (fit, evaluate, save, route)
│   ├── mlp.py                  # TabularMLP architecture + training loop
│   ├── stage2.py               # Stage 2 meta-learning orchestration (~1680 lines)
│   ├── metrics.py              # All evaluation metrics (gold + weak + ranking)
│   ├── synthetic_augment.py    # Support-set augmentation (noise/mixup/SMOTE-NC)
│   ├── episode_sampler.py      # Few-shot episode construction + task tables
│   ├── anil.py                 # ANIL meta-learner
│   ├── FOMAML.py               # FOMAML meta-learner
│   └── maml.py                 # MAML meta-learner
├── experiments/
│   ├── stage1_config.yaml      # Stage 1 SINGLE SOURCE OF TRUTH config
│   ├── stage2_config.yaml      # Stage 2 base config + 17+ named presets
│   └── run_stage1.py           # CLI entrypoint for Stage 1 training
├── notebooks/
│   ├── 16_stage1_baselines.ipynb       # Stage 1 training + artifact saving
│   ├── 17_stage2_meta_learning.ipynb   # Stage 2 experiment runner
│   ├── 17.1_stage2_analysis_dashboard.ipynb  # Stage 2 results analysis
│   └── 19.0_synthetic_augmentation_experiment.ipynb  # Augmentation grid
├── models/
│   └── stage_1/
│       ├── global/A_weak_only/         # Condition A artifacts
│       └── Logistics/
│           ├── B_gold_only/            # Condition B artifacts
│           ├── C_hybrid/               # Condition C artifacts
│           └── D_hybrid_unweighted/    # Condition D artifacts
└── experiments/                        # Stage 2 run outputs
    └── {experiment_id}/
        ├── resolved_config.yaml
        ├── metrics.json
        └── predictions.parquet
```

**Confirmed on-disk Stage 1 artifacts (per metadata.json):** `model.pt`, `preprocessor.pkl`, `metadata.json`, `history.json`. The file `feature_names.json` is referenced in `save_stage1_condition_artifacts()` but **has NOT been confirmed present on disk** — this is a risk item.

**Confirmed Stage 2 experiment outputs exist** for multiple experiment IDs with deterministic naming: `stage2__init-{init}__method-{method}__kpos-{k}__kneg-{k}__steps-{s}__target-{dept}[__lr-{lr}]`.

---

## 2. Stage 1 Current Implementation

### Four Conditions

| Condition | Name | Training labels | Target-aware? | Routing |
|---|---|---|---|---|
| A | `A_weak_only` | Snorkel `renegotiation_prob` (all depts) | No (global) | `global/A_weak_only/` |
| B | `B_gold_only` | Gold `gold_y` (non-target depts only) | Yes | `{target}/B_gold_only/` |
| C | `C_hybrid` | Weak all + Gold non-target (weighted 1:5) | Yes | `{target}/C_hybrid/` |
| D | `D_hybrid_unweighted` | Weak all + Gold non-target (equal weight) | Yes | `{target}/D_hybrid_unweighted/` |

### Key Functions (`stage1.py`)

- `fit_all_stage1_models(df, target_department, config)` — orchestrates all four conditions
- `get_stage1_condition_dir(condition, target_department)` — canonical path routing; A goes to `global/A_weak_only`, B/C/D go to `{target_dept}/{condition}`
- `save_stage1_condition_artifacts(model, preprocessor, condition, target_department, history, tabular_results, feature_names)` — saves all artifacts; calls `get_stage1_condition_dir()` for routing
- `save_stage1_metadata(condition, target_department, feature_count, config_snapshot, n_train, n_val)` — saves `metadata.json`
- `evaluate_stage1_on_weak_labels()` / `evaluate_stage1_on_gold_labels()` — separate evaluation paths
- `evaluate_stage1_bundle()` — evaluates on both weak and gold
- `GLOBAL_CONDITIONS = {"A_weak_only"}` — used by stage2.py to route init loading

### CLI Training (`experiments/run_stage1.py`)

- `load_and_split()` — implements `gold_mask_train = (y_gold_train >= 0) & (dept_train != target_department)` — **confirmed correct target exclusion**
- `get_validation_targets()` — returns **weak labels for A_weak_only** (per YAML comment), **gold labels for B/C/D** — **confirmed correct**
- `build_conditions()` — builds all four condition dicts with correct label assignment and WeightedRandomSampler for C

### Target-Aware Exclusion

Confirmed in both `run_stage1.py` (CLI) and NB16 (notebook):
- Gold labels for the target department are **excluded from Stage 1 B/C/D training**
- NB16 cell 8: `df_gold_non_target = df_gold[df_gold["department"] != TARGET_DEPARTMENT]` with hard `ValueError` on leakage

### Known Inconsistency — NB16 vs CLI for Condition A Validation

In **NB16 cell 12**, ALL conditions (including A_weak_only) validate on `y_gold_test`. The CLI (`run_stage1.py → get_validation_targets()`) correctly returns **weak labels for A**. The YAML comment explicitly says *"A_weak_only validates on WEAK labels"*. **NB16 is inconsistent with both the CLI and the YAML.** If NB16 was used for the actual on-disk artifacts, the A_weak_only validation curve/early stopping may have been computed against gold labels, which could affect the reported validation loss and early stopping point — but not training signal (still weak labels).

---

## 3. Stage 1 MLP

### Architecture (`src/master_thesis/mlp.py`)

```
Input(154) → Linear(input→h1) → BatchNorm1d → ReLU → Dropout(p)
           → Linear(h1→h2) → BatchNorm1d → ReLU → Dropout(p)
           → Linear(h2→1)
```

Output: single logit. Loss: `BCEWithLogitsLoss`. Optimizer: `Adam`.

### Key Classes and Functions

- `TabularMLP(nn.Module)` — defined above
- `MLPTrainingConfig` (dataclass) — defaults: `n_epochs=50, patience=5, lr=1e-3, weight_decay=1e-5, hidden_dim_1=128, hidden_dim_2=64, dropout=0.2`
- `fit_mlp_model(X_train, y_train, X_val, y_val, config, sample_weight)` — high-level trainer; uses `WeightedRandomSampler` when `sample_weight` is provided (condition C)
- `train_mlp(model, train_loader, val_loader, optimizer, criterion, config)` — inner loop with early stopping on val_loss; saves best `state_dict`
- `build_mlp(input_dim, config)` — constructs model + optimizer + criterion
- `predict_mlp_bundle(model, X)` — returns `sigmoid(logits)` as probabilities
- `GroupShuffleSplit` — used for reproducible train/val splits (grouped by contract_id to prevent leakage)

### Architecture Config Discrepancy (CRITICAL — see Section 9)

**On-disk metadata (confirmed):** `hidden_dim_1=128, hidden_dim_2=64, dropout=0.2`
**YAML default + Stage 2 resolved_config:** `hidden_dim_1=256, hidden_dim_2=128, dropout=0.3`
These are different architectures. The torch files were not inspectable in the sandbox (torch not available), so actual weight tensor shapes were not verified.

---

## 4. Stage 2 Current Implementation

### Key Functions (`src/master_thesis/stage2.py`, ~1680 lines)

**Routing and Discovery:**
- `stage1_artifact_dir(init_name, target_department)` — canonical routing: `A_weak_only → global/A_weak_only`, `B/C/D → {target}/{init}`, `random → ValueError`
- `discover_stage1_artifacts()` — scans disk, skips legacy flat dirs
- `validate_experiment_grid(grid)` — pre-flight: confirms `model.pt` and `preprocessor.pkl` exist for every `(init, target)` cell; raises if missing

**Experiment Resolution:**
- `resolve_stage2_paths(experiment_id, ...)` — all filesystem paths for one run
- `resolve_experiment_grid(config)` — cartesian product: `methods × support_grid × inner_steps × init_names × target_departments [× lr] [× synthetic]`
- `generate_experiment_id(...)` — deterministic ID string

**Task Preparation:**
- `prepare_stage2_tasks(df, target_department, ...)` — builds task table, calls `filter_valid_departments()`, creates target episodes, **ASSERTS `target_department not in meta_train_departments`** — confirmed leakage guard exists

**Model Loading and Building:**
- `load_stage1_model_weights(init_name, target_department, model)` — `strict=True` state_dict loading
- `build_stage2_model_from_config(input_dim, config)` — builds `TabularMLP` using `stage2_config.yaml`'s `mlp_config` section (256/128)
- `backfill_legacy_stage2_results()` — migrates old flat-layout runs

**Meta-Learning Dispatching:**
- `run_stage2_method(method, model, ...)` — dispatches to `zero_shot / finetune / anil / fomaml / maml`

### Constants

```python
GLOBAL_INIT_NAMES = ("A_weak_only",)
TARGET_AWARE_INIT_NAMES = ("B_gold_only", "C_hybrid", "D_hybrid_unweighted")
RANDOM_INIT_NAME = "random"
```

### Meta-Learners

| File | Functions |
|---|---|
| `anil.py` | `meta_train_anil()`, `evaluate_anil_on_target_episodes()`, `adapt_anil_on_episode()` |
| `FOMAML.py` | `meta_train_fomaml()`, `evaluate_fomaml_on_target_episodes()`, `adapt_fomaml_on_episode()` |
| `maml.py` | `meta_train_maml()`, `evaluate_maml_on_target_episodes()`, `adapt_maml_on_episode()` |

### Stage 2 Config Presets (17+ in `stage2_config.yaml`)

Selected important presets:
- `thesis_main` — primary benchmark run
- `quick_debug` — fast smoke test
- `four_method_benchmark` / `five_method_benchmark` — method comparison
- `k_shot_sweep` — k-shot ablation
- `lr_sweep_benchmark` — inner LR ablation
- `initialization_ablation` / `full_init_ablation` — init comparison
- `synthetic_main_logistics` — augmentation grid (used by NB19)

Default hyperparameters: `n_repeats=20, meta_iterations=200, meta_batch_size=2, inner_lr=0.01, outer_lr=0.001`

---

## 5. Evaluation and Metrics

### Weak-Label Evaluation (`metrics.py → evaluate_predictions()`)
- RMSE, MAE, R² — comparing predicted probability to `renegotiation_prob`

### Gold-Label Evaluation (`metrics.py → evaluate_on_gold_binary()`)
- **Discrimination:** AUROC, Average Precision (AP)
- **Calibration:** ECE (10-bin, `compute_ece()`), Brier score
- **Threshold-based:** F1, accuracy, log-loss
- **Ranking:** Precision@K, Recall@K, NDCG@K for K ∈ {5, 10, 20}

### Per-Department Evaluation
- `evaluate_grouped_gold_binary()` — runs `evaluate_on_gold_binary()` per department group

### SHAP / Interpretability
- `calculate_shap_divergence()` — "brain shift" metric: total variation distance between SHAP distributions + top-5 feature Jaccard similarity

### Notebook 17.1 Analysis
- Loads `experiment_summary.csv` (experiment registry)
- Enriches with artifact metadata
- Produces AUROC/AP/NDCG@K plots, k-shot curves, method comparisons, init ablation charts
- Includes Stage 1 artifact routing audit at the top

---

## 6. Config and Reproducibility

### Stage 1 Config (`experiments/stage1_config.yaml`)
- `seed=42`, `test_size=0.30`, `target_department="Logistics"`
- MLP: `n_epochs=100, patience=10, lr=5e-4, weight_decay=1e-4, train_batch_size=256, hidden_dim_1=256, hidden_dim_2=128, dropout=0.1`
- `hybrid_weak_weight=1.0, hybrid_gold_weight=5.0`
- Labeled as "SINGLE SOURCE OF TRUTH"

### Stage 2 Config (`experiments/stage2_config.yaml`)
- `base_config.mlp_config: hidden_dim_1=256, hidden_dim_2=128, dropout=0.3`
- Random init uses `random_preprocessor_source: "A_weak_only"` (borrows preprocessor from A)
- Each experiment writes `resolved_config.yaml` to its directory for full reproducibility

### Actual On-Disk Stage 1 Architecture (confirmed via metadata.json)
- `hidden_dim_1=128, hidden_dim_2=64, dropout=0.2`
- This matches `MLPTrainingConfig` **dataclass defaults**, NOT the YAML

### Seed Usage
- `SEED=42` in `config.py`
- `GroupShuffleSplit(random_state=SEED)` for train/val splits
- Episodic sampling uses `random_state` parameters throughout

### Experiment ID Determinism
- `generate_experiment_id()` produces the same ID for the same hyperparameter combination — safe to re-run and check if output already exists

---

## 7. Data and Feature Assumptions

### Input DataFrame
- Column `renegotiation_prob` — Snorkel-distilled weak label (float, [0,1])
- Column `gold_y` — binary gold label (int: 0/1, -1 for unlabeled)
- Column `department` — department string (used for target-aware splitting and episode construction)
- Column `contract_id` — unique contract identifier (used for GroupShuffleSplit and episode construction)
- Column `label_source` — used in augmented rows (`"SYNTHETIC_{METHOD}"`)
- Column `is_synthetic` — boolean flag set to True in augmented support rows

### Feature Count
- **154 features** confirmed via both A_weak_only and B_gold_only metadata.json
- Feature preprocessing handled by a fitted `preprocessor.pkl` (sklearn pipeline, likely ColumnTransformer)

### Gold Label Filtering
- `gold_mask = (gold_y >= 0)` — unlabeled rows excluded from gold training
- `gold_mask &= (department != target_department)` — target dept excluded from B/C/D training

### Target Department
- Default throughout: `"Logistics"`
- Configurable in both YAML and notebook variables

### Train/Val Split
- `test_size=0.30` (from YAML)
- `GroupShuffleSplit` grouped by `contract_id` — prevents contract-level leakage across split

---

## 8. Synthetic Support Augmentation

### Public API (`synthetic_augment.py → augment_support()`)
- Accepts: `"none"`, `"gaussian_noise"`, `"mixup"`, `"smote_nc"`
- Generates synthetic rows **per class separately** (maintains class balance control)
- All synthetic rows tagged: `is_synthetic=True`, `label_source="SYNTHETIC_{METHOD}"`, `contract_id="SYNTH_{uuid4}"`
- Hard error if `department` column changes during augmentation

### SMOTE-NC Implementation (`_smote_nc()`)
- **Option B (same-contract exclusion):** interpolation only between rows from *different* contracts
- Falls back to `gaussian_noise` if fewer than 2 unique contracts per class

### Integration with Episode Sampler (`episode_sampler.py → sample_episode_with_synthetic_support()`)
- Pipeline order: (1) real support/query split first, (2) augment support only, (3) assert no synthetic IDs in query set
- `filter_valid_departments()` — filters for departments that can supply `n_support_pos + n_support_neg + ≥1 query contracts`
- `make_logistics_meta_test_split()` — creates `n_repeats` episodes for target department
- `build_department_task_table()` — contract-level gold-label task table

### Notebook Driver
- NB19 (`19.0_synthetic_augmentation_experiment.ipynb`) — reads `synthetic_*` presets from stage2_config.yaml, drives the full augmentation grid

---

## 9. Most Important Bugs or Risks (Ranked)

### 🔴 CRITICAL — MLP Architecture Mismatch Between Stage 1 and Stage 2

**What:** Stage 1 on-disk artifacts have `hidden_dim_1=128, hidden_dim_2=64, dropout=0.2` (confirmed via metadata.json for both A_weak_only and B_gold_only). Stage 2 `build_stage2_model_from_config()` builds the model with `hidden_dim_1=256, hidden_dim_2=128, dropout=0.3` (confirmed in all resolved_config.yaml files inspected).

**Why this matters:** `load_stage1_model_weights()` uses `strict=True`. If the model was built with 256/128 dims but the `.pt` file contains 128/64 weight tensors, the load would **fail with a shape mismatch error**. However, experiments exist on disk and apparently completed successfully. This means either: (a) the `.pt` files actually contain 256/128 weights (Stage 1 was trained with the YAML's config, not the dataclass defaults, and metadata.json was written stale/wrong), or (b) there's a code path that adapts the architecture we haven't found, or (c) the experiments on disk were run with `random` init or otherwise didn't load Stage 1 weights.

**What to do:** Load the actual `.pt` file and inspect weight tensor shapes: `torch.load('model.pt', map_location='cpu')` — check `linear1.weight.shape`. If it's `(256, 154)`, Stage 1 was trained with 256/128 and metadata.json is wrong. If it's `(128, 154)`, completed Stage 2 experiments either crashed silently or used random init.

---

### 🔴 CRITICAL — NB16 Validation Target Inconsistency for Condition A

**What:** NB16 cell 12 uses `y_val=y_gold_test` for ALL conditions including A_weak_only. The CLI (`run_stage1.py`) correctly uses weak labels for A and gold for B/C/D. The YAML comment explicitly states the correct behavior.

**Why this matters:** If NB16 produced the on-disk artifacts (likely), condition A's early stopping and val_loss curves were computed against gold labels, not weak labels. This affects the trained model's behavior because early stopping may have fired at a different epoch than intended.

**What to do:** Re-run Stage 1 condition A via CLI (`run_stage1.py`) with the YAML config to get a correctly-validated A artifact. Check whether the re-trained model produces meaningfully different downstream Stage 2 results.

---

### 🟠 HIGH — `feature_names.json` Not Confirmed On Disk

**What:** `save_stage1_condition_artifacts()` documents saving `feature_names.json` alongside `model.pt`, `preprocessor.pkl`, `metadata.json`, and `history.json`. The file was not found/confirmed in the actual artifact directories.

**Why this matters:** Any downstream analysis that tries to load feature names for SHAP or interpretability will fail silently or KeyError.

**What to do:** Check `ls models/stage_1/global/A_weak_only/` and `ls models/stage_1/Logistics/B_gold_only/`. If absent, add a one-time save call or re-run Stage 1 with a patched save function.

---

### 🟠 HIGH — Legacy Flat Stage 1 Artifact Directories

**What:** `discover_stage1_artifacts()` explicitly skips legacy flat directories (the old layout before the canonical target-aware layout was introduced). These directories may exist on disk and could confuse manual inspection or old notebooks.

**Why this matters:** If any code path still looks in the flat layout, it will find stale artifacts. If any notebook has a hard-coded flat path, it will silently load the wrong model.

**What to do:** Run `discover_stage1_artifacts()` and compare what it finds to a manual `find models/stage_1/ -name 'model.pt'`. Confirm only canonical paths produce hits.

---

### 🟡 MEDIUM — Stage 1 MLP Config Defaults vs YAML Divergence

**What:** `MLPTrainingConfig` dataclass defaults (`n_epochs=50, patience=5, lr=1e-3, weight_decay=1e-5, hidden_dim_1=128, hidden_dim_2=64, dropout=0.2`) differ from `stage1_config.yaml` (`n_epochs=100, patience=10, lr=5e-4, weight_decay=1e-4, hidden_dim_1=256, hidden_dim_2=128, dropout=0.1`). If any call to `fit_mlp_model()` or `build_mlp()` omits a config argument and falls back to dataclass defaults, it silently trains the wrong architecture.

**What to do:** Audit all call sites of `fit_mlp_model()` and `build_mlp()` to confirm they all receive an explicit `MLPTrainingConfig` constructed from the YAML. Add an assertion in `fit_all_stage1_models()` that the config was loaded from YAML, not constructed with bare defaults.

---

### 🟡 MEDIUM — Stage 2 `resolved_config.yaml` Dropout Mismatch

**What:** Stage 2 experiments use `dropout=0.3`, Stage 1 artifacts used `dropout=0.2` (per metadata). At init-loading time, if the Stage 2 model is built with 0.3 dropout and loaded with Stage 1 weights (trained at 0.2 dropout), the model runs but with a different dropout rate in fine-tuning vs pre-training. This is not a shape mismatch (dropout doesn't affect weight shapes) but affects training dynamics.

**What to do:** Decide intentionally whether to inherit Stage 1's dropout for fine-tuning or use a separate Stage 2 dropout. Document the choice.

---

### 🟡 MEDIUM — `backfill_legacy_stage2_results()` Side Effects

**What:** This function migrates old flat-layout Stage 2 results into the `experiments/` directory structure. If called multiple times, it may duplicate results. Its triggering conditions are not fully clear.

**What to do:** Audit whether this function is called automatically in any notebook or at import time, and add idempotency guards.

---

## 10. Recommended Next Implementation Steps (Ordered Checklist)

1. **[URGENT] Resolve architecture mismatch** — load `models/stage_1/global/A_weak_only/model.pt` with torch and print `model.linear1.weight.shape`. This single check definitively resolves whether all on-disk experiments are valid.

2. **[URGENT] Re-run Stage 1 via CLI** — use `experiments/run_stage1.py` with `experiments/stage1_config.yaml` as the authoritative training run. This ensures: correct validation targets for A, correct YAML-derived architecture, reproducible seed behavior.

3. **[URGENT] Confirm `feature_names.json` presence** — `ls models/stage_1/global/A_weak_only/` and patch `save_stage1_condition_artifacts()` if absent.

4. **Audit legacy flat directories** — run `discover_stage1_artifacts()`, compare to manual find, delete or archive stale flat-layout dirs.

5. **Audit all `fit_mlp_model()` call sites** — ensure every call passes an explicit YAML-derived `MLPTrainingConfig`, never relying on dataclass defaults.

6. **Lock dropout intentionally** — decide if Stage 2 should inherit Stage 1 dropout (0.2) or use its own (0.3). Document and enforce in `build_stage2_model_from_config()`.

7. **Add NB16 cell 12 fix** — change `y_val=y_gold_test` for condition A to `y_val=y_weak_val` to align with CLI and YAML intent.

8. **Run `validate_experiment_grid()`** — before any new Stage 2 run, confirm all `(init, target)` cells are populated. This function exists but should be the first cell in NB17.

9. **Verify `experiment_summary.csv` completeness** — confirm all runs in `experiments/` are registered in the experiment registry. `backfill_legacy_stage2_results()` should cover old runs but audit manually.

10. **Add end-to-end smoke test** — `quick_debug` preset in Stage 2 config is designed for this. Run NB17 with `PRESET = "quick_debug"` and confirm: artifact loads correctly, no shape errors, metrics write to disk, NB17.1 can load and plot the result.

11. **Write thesis results section** — once architecture mismatch is resolved and CLI Stage 1 re-run is complete, NB17.1 dashboard is the primary source for AUROC/AP/NDCG@K comparison tables (methods × init × k-shot).

---

## 11. Memory-Ready Summary

**System:** Binary classification of contract renegotiation likelihood. Two-stage pipeline: Stage 1 trains a TabularMLP initializer on historical data; Stage 2 fine-tunes it to a specific target department using meta-learning (few-shot).

**Stage 1 — Four Conditions:**
- A (`A_weak_only`): Trains on all departments using Snorkel weak labels (`renegotiation_prob`). Global model. Artifact at `models/stage_1/global/A_weak_only/`.
- B (`B_gold_only`): Trains on gold labels (`gold_y`) excluding target department. Target-aware. Artifact at `models/stage_1/{target}/B_gold_only/`.
- C (`C_hybrid`): Weak labels (all depts) + gold labels (non-target depts), gold upweighted 5×. Uses `WeightedRandomSampler`. Artifact at `models/stage_1/{target}/C_hybrid/`.
- D (`D_hybrid_unweighted`): Same as C but equal weights. Artifact at `models/stage_1/{target}/D_hybrid_unweighted/`.
- Target exclusion enforced at training: no target-dept gold labels in B/C/D.

**Stage 1 MLP:** `TabularMLP` — 2-layer feedforward with BatchNorm + Dropout + BCEWithLogitsLoss. Input: 154 features. On-disk: `128→64→1`. YAML config says `256→128→1`. **This discrepancy is unresolved — see Bug #1.**

**Stage 2 — Meta-Learning:**
- Methods: `zero_shot`, `finetune`, `ANIL`, `FOMAML`, `MAML`
- Target department: `Logistics` (default)
- Episodes: few-shot (k_pos/k_neg support contracts + query contracts), constructed at contract level
- Init: one of `A_weak_only / B_gold_only / C_hybrid / D_hybrid_unweighted / random`
- `load_stage1_model_weights()` uses `strict=True` — architecture must match exactly
- `build_stage2_model_from_config()` reads from `stage2_config.yaml` mlp_config section
- Pre-flight: `validate_experiment_grid()` checks all `(init, target)` artifact cells exist
- Leakage guard: `assert target_department not in meta_train_departments` in `prepare_stage2_tasks()`

**Synthetic Augmentation:**
- `augment_support()`: `gaussian_noise / mixup / smote_nc / none`
- Applied to support set only, after real support/query split, before meta-training
- SMOTE-NC uses cross-contract interpolation; falls back to gaussian if <2 contracts per class
- Synthetic rows tagged with `is_synthetic=True` and `contract_id="SYNTH_{uuid}"`
- Query set purity asserted: no synthetic IDs in query

**Metrics:**
- Weak: RMSE, MAE, R²
- Gold: AUROC, AP, ECE, Brier, F1, accuracy, Precision@K, Recall@K, NDCG@K (K=5,10,20)
- Brain shift: SHAP total variation distance + top-5 Jaccard

**Critical Open Issues (as of 2026-04-26):**
1. **Architecture mismatch** — Stage 1 metadata says 128/64; Stage 2 config says 256/128. Actual `.pt` tensor shapes not yet verified. All completed experiments may be invalid if Stage 2 could not load Stage 1 weights.
2. **NB16 condition A validation uses gold labels** — contradicts CLI and YAML intent. On-disk A artifact may have been early-stopped on wrong signal.
3. **`feature_names.json` unconfirmed on disk** — needed for SHAP analysis.

**Key Files to Know:**
- `src/master_thesis/stage1.py` — Stage 1 orchestration + artifact routing
- `src/master_thesis/stage2.py` — Stage 2 orchestration (~1680 lines), all routing and experiment management
- `src/master_thesis/mlp.py` — TabularMLP definition and training
- `experiments/stage1_config.yaml` — authoritative Stage 1 config
- `experiments/stage2_config.yaml` — Stage 2 config + 17+ presets
- `experiments/run_stage1.py` — correct CLI entrypoint for Stage 1
- `notebooks/17_stage2_meta_learning.ipynb` — Stage 2 runner
- `notebooks/17.1_stage2_analysis_dashboard.ipynb` — results analysis

**Seed:** `SEED=42` throughout. `GroupShuffleSplit` for contract-level split. Deterministic experiment IDs via `generate_experiment_id()`.

**Target department for all confirmed runs:** `Logistics`.

---
*End of handoff document.*
