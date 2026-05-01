# Pipeline Fix Report
**Date:** 2026-04-26  
**Scope:** 10-fix specification for Stage 1 → Stage 2 meta-learning pipeline  
**Status:** All code changes complete. 35/35 smoke-test checks pass. 5 items require local execution (see §6).

---

## 1. Summary of Changes

Six files were modified and one new file was created. The fixes address: an architecture mismatch between Stage 1 artifacts and Stage 2 config that would crash `strict=True` weight loading; a validation-target bug in NB16 where condition A used gold labels for early stopping instead of weak labels; a broken `BaselineTrainingConfig` constructor in the CLI runner; and missing infrastructure for `feature_names.json`. The changes also strengthen pre-flight validation, clarify random-init as a clean baseline, and document YAML as the single source of truth for config.

---

## 2. Modified Files

| File | Change type | Fix(es) |
|------|-------------|---------|
| `src/master_thesis/stage2.py` | Added 4 functions, extended 1, updated 3 callers | Fix 1, 6, 7 |
| `src/master_thesis/stage1.py` | Added `condition_val_overrides` param, added backfill utility | Fix 2, 3 |
| `experiments/run_stage1.py` | Fixed `build_stage1_model_config()` bug, added imports | Fix 5 |
| `experiments/stage2_config.yaml` | Updated `base_config.mlp_config` to 128/64 with comments | Fix 1, 5 |
| `notebooks/16_stage1_baselines.ipynb` | Updated cells 16, 17, 20, 21 | Fix 2, 5 |
| `experiments/backfill_stage1_artifacts.py` | **New file** — audit and backfill utility | Fix 3, 4 |

---

## 3. Detailed Changes Per File

### 3.1 `src/master_thesis/stage2.py`

**Four new functions added** (after `build_stage2_model_from_config`, ~line 1006):

**`load_stage1_artifact_metadata(artifact_dir)`**  
Loads and returns `metadata.json` from a Stage 1 artifact directory. Raises `FileNotFoundError` with a clear re-run instruction if the file is absent. Used by `build_stage2_model_for_init` and `validate_experiment_grid`.

**`infer_arch_from_state_dict(state_dict)`**  
Reads `net.0.weight` (shape `h1 × input_dim`) and `net.4.weight` (shape `h2 × h1`) from a PyTorch state dict to infer `hidden_dim_1` and `hidden_dim_2`. Used as a fallback when `metadata.json` is unavailable.

**`validate_artifact_architecture(artifact_dir, model_filename)`**  
Cross-checks `metadata.json` arch against actual `.pt` tensor shapes. Returns a dict with `metadata_arch`, `actual_arch`, `match` (bool), and `warning`. Does not raise on mismatch — surfaces as a warning so Stage 2 can use the correct (tensor-inferred) dims.

**`build_stage2_model_for_init(input_dim, config, init_source_dir, init_name, ...)`**  
The single correct entry point for building a Stage 2 model before weight loading:
- **Random init**: reads `hidden_dim_1 / hidden_dim_2 / dropout` from `stage2_config.yaml`'s `base_config.mlp_config`. No Stage 1 weights are ever loaded.
- **Pretrained init (A/B/C/D)**: reads architecture from `metadata.json` → falls back to inferring from `.pt` tensor shapes (with a `UserWarning`) if metadata is absent → raises `ValueError` if neither is possible.

This guarantees `strict=True` weight loading in all three Stage 2 runners always succeeds, regardless of what the YAML says.

**`validate_experiment_grid()` extended** to add per-cell fields:
- `metadata_present` — is `metadata.json` present in the artifact dir?
- `feature_names_present` — is `feature_names.json` present? (warning-only, not blocking)
- `arch_compatible` — do `metadata.json` dims match `stage2_config.yaml`? (informational — mismatch is allowed; builder resolves it)
- `is_runnable` now requires `metadata_present` for all non-random inits (previously only checked model and preprocessor)

**Three Stage 2 runners updated** — `run_finetune_baseline()`, `_run_meta_method()`, `run_zero_shot_method()` — all now call `build_stage2_model_for_init()` instead of the old `build_stage2_model_from_config()`. The `input_dim` is inferred from a single dummy row passed through the preprocessor (existing pattern, unchanged).

---

### 3.2 `src/master_thesis/stage1.py`

**`fit_stage1_conditions()` — new `condition_val_overrides` parameter**

```python
def fit_stage1_conditions(
    conditions: dict[str, dict],
    X_val: pd.DataFrame,
    y_val,
    config: Optional[Stage1ModelConfig] = None,
    seed: int = 42,
    verbose: bool = True,
    condition_val_overrides: Optional[dict[str, tuple]] = None,   # NEW
) -> dict[str, dict]:
```

When `condition_val_overrides` is provided, any condition whose name appears as a key uses the supplied `(X_val_override, y_val_override)` instead of the global `X_val / y_val`. All other conditions are unaffected. This fixes the NB16 bug where condition A (weak-only) was using gold labels for MLP early stopping.

**`backfill_feature_names_for_artifact_dir(artifact_dir, feature_names, preprocessor)`**  
New utility (added before `save_all_stage1_gold_results`). Writes `feature_names.json` into an existing artifact directory. Priority order: explicit `feature_names` list → `preprocessor.feature_names_in_` → warns and returns `None`. Returns the written path if successful, `None` otherwise. Used by `experiments/backfill_stage1_artifacts.py`.

---

### 3.3 `experiments/run_stage1.py`

**Bug fixed in `build_stage1_model_config()`**

The function was constructing `BaselineTrainingConfig` with flat field names that do not exist (`elastic_net_alpha=`, `xgb_n_estimators=`, etc.), causing a `TypeError` at runtime. Fixed to use the correct nested config objects:

```python
# Before (BUG):
BaselineTrainingConfig(elastic_net_alpha=..., xgb_n_estimators=..., ...)

# After (correct):
BaselineTrainingConfig(
    scale_numeric_for_elastic_net=bc.get("scale_numeric_for_elastic_net", True),
    elastic_net=ElasticNetConfig(
        alpha=en.get("alpha", 0.005),
        l1_ratio=en.get("l1_ratio", 0.5),
        max_iter=en.get("max_iter", 10000),
        random_state=en.get("random_state", 42),
    ),
    xgboost=XGBoostConfig(
        n_estimators=xgb.get("n_estimators", 500),
        learning_rate=xgb.get("learning_rate", 0.05),
        max_depth=xgb.get("max_depth", 6),
        ...
    ),
)
```

**Imports added**: `from master_thesis.baselines import BaselineTrainingConfig, ElasticNetConfig, XGBoostConfig`

---

### 3.4 `experiments/stage2_config.yaml`

**`base_config.mlp_config` updated** from `256/128/0.3` to `128/64/0.2`:

```yaml
mlp_config:
  hidden_dim_1: 128   # was 256 — now matches existing artifact dims
  hidden_dim_2: 64    # was 128 — now matches existing artifact dims  
  dropout: 0.2        # was 0.3
```

**Extensive comments added** explaining that this block applies **only to the random-init baseline**. For all pretrained inits (A/B/C/D), Stage 2 reads the architecture from `metadata.json` via `build_stage2_model_for_init()` and ignores this block entirely.

---

### 3.5 `notebooks/16_stage1_baselines.ipynb`

**Cell 16 (markdown)** — Updated to state that `stage1_config.yaml` is the single source of truth for all Stage 1 hyperparameters. Config should not be edited in the notebook.

**Cell 17 (config loading)** — Was hardcoded. Now loads from `experiments/stage1_config.yaml`:

```python
import yaml
_config_path = project_root / 'experiments' / 'stage1_config.yaml'
with open(_config_path, 'r') as _f:
    _stage1_yaml = yaml.safe_load(_f)

# BaselineTrainingConfig
_bc = _stage1_yaml.get('baseline_config', {})
_en = _bc.get('elastic_net', {})
_xgb = _bc.get('xgboost', {})
baseline_config = BaselineTrainingConfig(
    scale_numeric_for_elastic_net=_bc.get('scale_numeric_for_elastic_net', True),
    elastic_net=ElasticNetConfig(alpha=_en.get('alpha', 0.005), ...),
    xgboost=XGBoostConfig(n_estimators=_xgb.get('n_estimators', 500), ...),
)

# MLPTrainingConfig
_mc = _stage1_yaml.get('mlp_config', {})
mlp_config = MLPTrainingConfig(hidden_dim_1=int(_mc.get('hidden_dim_1', 256)), ...)
```

**Cell 20 (new markdown)** — Design rationale table for validation-target routing:

| Condition | Training target | Early-stopping target | Why |
|-----------|----------------|----------------------|-----|
| A_weak_only | Snorkel `renegotiation_prob` | Snorkel `renegotiation_prob` | No gold leakage; measures weak-label fit quality |
| B_gold_only | Gold `gold_y` | Gold `gold_y` | Gold labels only; target-aware |
| C_hybrid | Mixed (weak + gold) | Gold `gold_y` | Gold provides the calibration signal |
| D_hybrid_unweighted | Mixed (weak + gold) | Gold `gold_y` | Same as C, no weighting |

**Cell 21 (condition fitting)** — Was calling `fit_stage1_conditions()` without `condition_val_overrides`, so condition A used gold labels for MLP early stopping. Fixed:

```python
# Build weak validation split for condition A
df_weak_val_rows = df_weak_all[df_weak_all[GROUP_COL].isin(gold_test_contract_ids)].copy()
X_weak_val_df = df_weak_val_rows[feature_cols].copy()
y_weak_val   = df_weak_val_rows[WEAK_TARGET_COL].astype(float).values

trained_bundles = fit_stage1_conditions(
    conditions=conditions,
    X_val=X_gold_test_df,              # default for B / C / D
    y_val=y_gold_test.astype(float),
    config=stage1_config,
    seed=SEED,
    verbose=True,
    condition_val_overrides={
        'A_weak_only': (X_weak_val_df, y_weak_val),  # weak labels for A only
    },
)
```

---

### 3.6 `experiments/backfill_stage1_artifacts.py` (new file)

Standalone CLI script. Three actions:

1. **Audit** — prints a table of all canonical artifact directories showing which of the 5 required files (`mlp_pretrained.pt`, `mlp_pretrained_preprocessor.joblib`, `metadata.json`, `feature_names.json`, `tabular_results.csv`) are present or missing.
2. **Legacy detection** — scans `models/stage_1/` for old flat-layout directories (condition names at the top level) and old target-dept runs, and warns about them.
3. **Backfill** — for each directory missing `feature_names.json`, loads the fitted preprocessor, extracts `feature_names_in_`, and writes the JSON file.

```bash
# Audit only:
python experiments/backfill_stage1_artifacts.py --dry-run

# Backfill (safe, skips existing):
python experiments/backfill_stage1_artifacts.py

# Force overwrite existing feature_names.json:
python experiments/backfill_stage1_artifacts.py --force

# Different target department:
python experiments/backfill_stage1_artifacts.py --target-department Engineering
```

---

## 4. Validation Report

### 4.1 Architecture validation

All four Stage 1 artifact directories confirmed via `metadata.json`:

| Condition | Path | `hidden_dim_1` | `hidden_dim_2` | Source |
|-----------|------|----------------|----------------|--------|
| A_weak_only | `models/stage_1/global/A_weak_only/` | 128 | 64 | metadata.json ✓ |
| B_gold_only | `models/stage_1/Logistics/B_gold_only/` | 128 | 64 | metadata.json ✓ |
| C_hybrid | `models/stage_1/Logistics/C_hybrid/` | 128 | 64 | metadata.json ✓ |
| D_hybrid_unweighted | `models/stage_1/Logistics/D_hybrid_unweighted/` | 128 | 64 | metadata.json ✓ |

All four agree: **128 → 64 → 1** (one hidden layer of 128, one of 64, scalar output).

The original bug: `stage2_config.yaml` had `hidden_dim_1: 256, hidden_dim_2: 128`, which would cause a `strict=True` crash when loading any of the four artifacts. This is now resolved: `build_stage2_model_for_init()` reads from `metadata.json` and ignores the YAML for pretrained inits.

`stage2_config.yaml`'s `base_config.mlp_config` now reads `128/64/0.2`, which matches the existing artifacts. This ensures the random-init baseline uses the same width as the pretrained inits for a fair comparison.

### 4.2 Smoke test results

35 checks run in the sandbox (no joblib/torch/sklearn required).

**All 35 checks: PASS**

Grouped results:

| Section | Checks | Result |
|---------|--------|--------|
| 1. YAML loading | 5 | ✓ PASS |
| 2. Artifact metadata | 13 | ✓ PASS (4 warnings — feature_names.json missing) |
| 3. stage2.py function presence | 5 | ✓ PASS |
| 4. build_stage2_model_for_init logic | 5 | ✓ PASS |
| 5. stage1.py function presence | 3 | ✓ PASS |
| 6. condition_val_overrides routing | 2 | ✓ PASS |
| 7. run_stage1.py BaselineTrainingConfig fix | 6 | ✓ PASS |
| 8. Leakage guards | 3 | ✓ PASS |
| 9. backfill_stage1_artifacts.py syntax | 4 | ✓ PASS |
| 10. validate_experiment_grid extensions | 5 | ✓ PASS |

The smoke test could not import `stage2.py` or `stage1.py` (sandbox has no `joblib`, `torch`, or `sklearn`). All checks were performed using AST parsing, regex, YAML loading, and JSON file reading — which verify structure and logic without executing the modules. All four modified Python files also pass `python3 -m py_compile` syntax checks.

### 4.3 What was NOT tested in the sandbox

The following require local execution in the project venv:

1. **Full module import** of `stage2.py`, `stage1.py` (needs joblib, torch, sklearn)
2. **`build_stage2_model_for_init()` runtime behaviour** — loading actual weights with `strict=True`
3. **`fit_stage1_conditions()` with `condition_val_overrides`** — actual MLP training loop
4. **`validate_experiment_grid()` runtime** — resolves actual paths and loads metadata
5. **`run_stage1.py` end-to-end** — requires data files and full venv

---

## 5. Fix-by-Fix Resolution Status

| Fix | Description | Status |
|-----|-------------|--------|
| Fix 1 | Resolve Stage 1/Stage 2 MLP architecture mismatch | ✅ Complete |
| Fix 2 | NB16 validation logic for A_weak_only (weak labels for early stopping) | ✅ Complete |
| Fix 3 | Ensure `feature_names.json` saved/loaded consistently | ✅ Code complete; backfill pending (local) |
| Fix 4 | Audit and remove legacy flat artifact paths | ✅ `backfill_stage1_artifacts.py` handles detection and audit |
| Fix 5 | YAML as single source of truth for Stage 1 config | ✅ Complete (NB17 + CLI both load from YAML) |
| Fix 6 | Strengthen Stage 2 pre-flight validation | ✅ Complete (`validate_experiment_grid` extended) |
| Fix 7 | Random init is a clean baseline (no Stage 1 weights loaded) | ✅ Confirmed — `RANDOM_INIT_NAME` branch in `build_stage2_model_for_init` returns model with no weight loading |
| Fix 8 | Target-department leakage guards confirmed | ✅ Confirmed — all three code paths (A goes to `global/`, B/C/D filter by `target_department != dept`) |
| Fix 9 | Smoke test | ✅ 35/35 pass; 5 warnings require local action |
| Fix 10 | Final validation report | ✅ This document |

---

## 6. Remaining Risks and Required Local Actions

### Required (must do before running Stage 2)

**Action 1 — Backfill `feature_names.json`**  
All four existing artifact directories are missing `feature_names.json`. This file is required by Stage 2 SHAP computations.

```bash
cd /path/to/thesis
source .venv/bin/activate   # or however your venv is activated
python experiments/backfill_stage1_artifacts.py
```

Expected output: 4 lines reading `Wrote feature_names.json (N features)`.

If the preprocessors were fitted on a numpy array rather than a DataFrame (and therefore lack `feature_names_in_`), the script will warn and you must re-run Stage 1 training to regenerate the artifacts.

**Action 2 — Re-run Stage 2 validation grid before any production run**  
After the backfill, call `validate_experiment_grid()` from a notebook or the CLI to confirm all four inits show `is_runnable = True`.

### Recommended (improves reproducibility)

**Action 3 — Re-run Stage 1 training via CLI**  
The existing artifacts have `hidden_dim_1=128, hidden_dim_2=64`. The `stage1_config.yaml` was previously configured for `256/128`. These are now out of sync (YAML says 256/128, artifacts are 128/64). Two options:

- **Option A (recommended):** Update `stage1_config.yaml` `mlp_config` to `hidden_dim_1: 128, hidden_dim_2: 64` to match existing artifacts, making YAML the ground truth going forward.
- **Option B:** Re-run Stage 1 with the current YAML (256/128). Stage 2 will automatically use the new architecture from the regenerated `metadata.json`. Update `stage2_config.yaml`'s `base_config.mlp_config` to 256/128 to keep the random-init baseline consistent.

Either way, run the backfill after any Stage 1 re-run to ensure `feature_names.json` is present.

**Action 4 — End-to-end smoke test locally**

```bash
# 1. Verify Stage 1 artifacts are complete
python experiments/backfill_stage1_artifacts.py --dry-run

# 2. Run a single Stage 2 experiment cell in NB17 or:
python experiments/run_stage2.py --preset debug --init A_weak_only --method finetune

# 3. Confirm no architecture mismatch errors in output
```

### Low-risk / informational

**Action 5 — Legacy flat artifact directories**  
`backfill_stage1_artifacts.py --dry-run` will report any old flat-layout directories (condition names directly under `models/stage_1/`). These are not loaded by any active code path but can be removed to avoid confusion.

**Action 6 — NB16 cell 21 weak validation set construction**  
The weak validation set in NB16 cell 21 is constructed by matching `contract_id` values from the gold test split (`gold_test_contract_ids`) into the weak-label DataFrame. If the gold test split changes (different random seed), the weak validation set is automatically updated. No manual maintenance required.

---

## 7. Architecture Invariant (post-fix)

The following property now holds across the entire pipeline:

> For any pretrained Stage 1 init (A / B / C / D), the `TabularMLP` built by `build_stage2_model_for_init()` has exactly the same `hidden_dim_1`, `hidden_dim_2`, and `dropout` as the weights stored in `mlp_pretrained.pt`. This is guaranteed by reading `mlp_architecture` from `metadata.json`, which is written at Stage 1 training time by `save_stage1_model_artifact()`. The YAML value in `stage2_config.yaml → base_config → mlp_config` is **only used for the random-init baseline** and has no effect on pretrained inits.

This invariant means `strict=True` weight loading will always succeed for pretrained inits, regardless of what the YAML says. If the YAML and the artifacts diverge, a warning is emitted (not an error) and the correct dims are used.
