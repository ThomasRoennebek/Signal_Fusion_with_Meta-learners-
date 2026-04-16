# Stage 2 Refactor Changelog

## Summary

This document records the root bugs fixed, files changed, and architectural
decisions made during the Stage 2 alignment pass. All changes preserve
existing filenames and do not break Stage 1 behavior.

---

## A. Root Problems Fixed

### 1. Notebook 17 — wrong import path for run_stage2

**Symptom:**
Cell 9 contained `from experiments.run_stage2 import run_experiments`.
The `experiments/` directory has no `__init__.py`, so this import always
raised `ModuleNotFoundError`.

**Fix:**
Cell 9 now uses `importlib.util.spec_from_file_location` to load
`run_stage2.py` from its absolute path at runtime. No `__init__.py` is
required, and the import is explicit about which file it loads.

Cell 1 also now adds both `str(SRC_PATH)` and `str(PROJECT_ROOT)` to
`sys.path` for robustness.

---

### 2. Notebook 17 — wrong `run_experiments` call signature

**Symptom:**
Cell 9 passed an unsupported `full_config=full_config` keyword argument to
`run_experiments`. The function signature is:
```python
run_experiments(experiment_list, skip_if_exists=True, force_rerun=False, verbose=True)
```

**Fix:**
Cell 9 now calls:
```python
run_experiments(
    experiment_list=experiment_list,
    skip_if_exists=not FORCE_RERUN,
    force_rerun=FORCE_RERUN,
    verbose=True,
)
```

---

### 3. Notebooks 17 and 17.1 — `group_col` vs `method_col` mismatch

**Symptom:**
Both notebooks called `plot_stage2_metric_comparison(..., group_col="method", ...)`
and `plot_stage2_metric_comparison(..., group_col="support_label", ...)`.
The function only accepted `method_col`, causing `TypeError: unexpected keyword
argument 'group_col'` on every plotting cell.

**Fix:**
`plot_stage2_metric_comparison` in `plotting.py` now accepts `group_col` as the
primary parameter name. The old `method_col` is kept as a deprecated alias via:
```python
if method_col is not None:
    group_col = method_col
```
This preserves backward compatibility while matching what both notebooks already
used.

---

### 4. Notebook 17.1 — `plot_inner_step_curve` called with `group_col`

**Symptom:**
Cell 12 called `plot_inner_step_curve(..., group_col="method", ...)`. The
function signature uses `method_col`, not `group_col`.

**Fix:**
Cell 12 now calls `method_col="method"` correctly.

---

### 5. Notebook 17.1 — `plot_repeated_episode_boxplot` called with `metrics=list`

**Symptom:**
Cell 15 passed a list to `metrics=`, but the function accepts a single string
via `metric_col=`.

**Fix:**
Cell 16 now iterates over the list and calls `plot_repeated_episode_boxplot`
once per metric.

---

### 6. Notebook 17.1 — `plot_prediction_probability_distribution` required `method_col`

**Symptom:**
The function called `_require_columns(predictions_df, [prob_col, method_col])`,
but a single-experiment predictions DataFrame typically has no "method" column,
causing a `ValueError` every time.

**Fix:**
`method_col` is now an optional parameter (`str | None = "method"`).
When `None` or absent from the DataFrame, the function plots a single
distribution without the hue grouping.

---

### 7. plotting.py — `VIEW_COLORS` undefined

**Symptom:**
`plot_view_level_shap_importance` referenced `VIEW_COLORS` which was never
defined, causing `NameError` whenever that function was called.

**Fix:**
`VIEW_COLORS` is now defined immediately after `METHOD_COLORS` with a sensible
default mapping for data view labels (financial, procurement, supplier, etc.).

---

## B. Files Changed

| File | Change type | Summary |
|------|------------|---------|
| `src/master_thesis/plotting.py` | Bug fix + feature | Add `VIEW_COLORS`; rename `method_col` -> `group_col` in `plot_stage2_metric_comparison` with backward-compat alias; make `method_col` optional in `plot_prediction_probability_distribution`; add `plot_support_size_curve` helper |
| `notebooks/17_stage2_meta_learning.ipynb` | Rewrite | Fix import, fix call signature, add artifact check cell, add quick summary table, add example override patterns in markdown |
| `notebooks/17.1_stage2_analysis_dashboard.ipynb` | Rewrite | Fix all API mismatches, add thesis-ready tables D–I, add reports/ output, add per-experiment artifact inspection improvements |
| `tests/test_episode_sampler.py` | Rewrite | Replace all placeholder tests with real contract-leakage, support-count, edge-case, and validity tests |
| `tests/test_stage2.py` | New file | Add tests for experiment grid resolution, ID uniqueness, preview DataFrame, config normalization, artifact schema, CLI runner preview-only mode |
| `STAGE2_REFACTOR_CHANGELOG.md` | New file | This file |

---

## C. What Was Standardized

### Experiment list format
`resolve_experiment_grid` returns `List[Tuple[str, Dict]]` consistently.
All consumers (notebook 17, `run_stage2.py`) unpack as `(exp_id, config)`.

### Artifact schema
Each completed experiment folder in `models/stage_2/experiments/<experiment_id>/`
is expected to contain:
- `resolved_config.yaml` — always written
- `metrics.csv` — written when `raw_metrics` is a non-empty DataFrame
- `predictions.csv` — written when `predictions` is a non-empty DataFrame
- `history.csv` — written when `history` is a non-empty DataFrame
- `stage2_result_summary.json` — always written (controls skip logic)
- `stage2_result.json` — always written (full serialized result)

### Plotting parameter names
- `plot_stage2_metric_comparison`: primary grouping parameter is now `group_col`
- `plot_prediction_probability_distribution`: `method_col` is optional
- `plot_inner_step_curve`: continues to use `method_col` (not `group_col`)

### Thesis table outputs
Notebook 17.1 now saves Tables D–I to `reports/stage2/`:
- `table_D_stage2_main_benchmark.csv`
- `table_E_stage2_adaptation_dynamics.csv`
- `table_F_stage2_support_size_sensitivity.csv`
- `table_G_stage2_init_ablation.csv`
- `table_H_stage2_leaderboard.csv`
- `table_I_stage1_vs_stage2_<department>.csv`

---

## D. What Still Requires Rerunning

The two existing experiment runs in `models/stage_2/experiments/` were produced
by a previous version and their artifacts may be complete. The following
scenarios require a fresh run:

1. **First-time full evaluation:** Run `thesis_main` preset to populate all
   method × init × support × inner-steps combinations.
2. **After any config change** to `stage2_config.yaml` that alters grid parameters:
   use `FORCE_RERUN = True` in notebook 17 or `--force-rerun` from CLI.
3. **Fine-tune anomaly:** The existing `finetune` run shows AUROC 0.50, which is
   suspicious. After verifying the data pipeline is intact, rerun with
   `FORCE_RERUN = True` for that experiment.

---

## E. How to Run

### Quick debug (CLI)

```bash
cd "experiments"
python run_stage2.py --preset quick_debug --preview-only
python run_stage2.py --preset quick_debug
```

### Full thesis sweep (CLI)

```bash
python run_stage2.py --preset thesis_main
```

### Force rerun specific experiments

```bash
python run_stage2.py --preset quick_debug --force-rerun
```

### From notebook 17

1. Open `notebooks/17_stage2_meta_learning.ipynb`
2. In the controls cell, set:
   ```python
   PRESET = "quick_debug"
   RUN_EXPERIMENTS = True
   ```
3. Run all cells.

### Analysis only (no rerun needed)

Open `notebooks/17.1_stage2_analysis_dashboard.ipynb` and run all cells.
This loads from `models/stage_2/experiments/experiment_summary.csv` only.

---

## F. Output Locations

| Artifact | Path |
|---------|------|
| Per-experiment artifacts | `models/stage_2/experiments/<experiment_id>/` |
| Master experiment registry | `models/stage_2/experiments/experiment_summary.csv` |
| Stage 1 model weights (initialization) | `models/stage_1/<init_name>/mlp_pretrained.pt` |
| Stage 1 preprocessor | `models/stage_1/<init_name>/mlp_pretrained_preprocessor.joblib` |
| Thesis-ready CSV tables (D–I) | `reports/stage2/` |

---

## G. Architecture Summary

```
experiments/stage2_config.yaml
    Central source of defaults, sweep grids, and named presets.

src/master_thesis/stage2.py
    Shared backend: config loading, grid resolution, data loading,
    task preparation, method dispatch, result standardization,
    artifact saving.

experiments/run_stage2.py
    CLI runner: loads config, resolves grid, calls run_experiments(),
    writes experiment_summary.csv.

src/master_thesis/{anil,FOMAML,maml}.py
    Method-specific training and evaluation. Each returns a standardized
    dict with status, raw_metrics, predictions, history.

src/master_thesis/episode_sampler.py
    Contract-aware support/query splitting. No support/query leakage.
    Exactly n_support_pos / n_support_neg positive/negative contracts.

notebooks/17_stage2_meta_learning.ipynb
    Thin controller: resolves grid, previews, optionally runs via
    run_experiments(), shows quick summary and artifact check.

notebooks/17.1_stage2_analysis_dashboard.ipynb
    Analysis dashboard: loads saved outputs, produces plots and
    thesis-ready tables D–I.
```
