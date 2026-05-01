# Stage 2 Experiment Structure Report

**Generated:** 2026-04-30  
**Scope:** Stage 2 meta-learning pipeline — YAML configuration, notebook roles, plot/table inventory, CLI and notebook usage guide

---

## 1. YAML Canonical Preset Groups

`experiments/stage2_config.yaml` is the **single source of truth** for all Stage 2 experiment parameters. There are exactly **10 canonical presets** in 4 groups. No experiment parameters are hardcoded in notebooks.

### Group 1 — Debug / Smoke Tests

| Preset | Methods | Departments | Purpose |
|--------|---------|-------------|---------|
| `quick_debug` | finetune, ANIL | Logistics | Fast smoke test: 2 repeats, 5 inner steps — verifies the full pipeline end-to-end without spending GPU compute |
| `stage2b_debug` | finetune, ANIL | Logistics | Stage 2b smoke test: confirms SMOTENC augmentation path writes support_accounting.csv and stage2b_config.json |

**When to use:** After any code change to `stage2.py`, `run_stage2.py`, or the YAML config. Takes ~1 min on CPU.

### Group 2 — Main Thesis Experiments

| Preset | Methods | Departments | Research Questions |
|--------|---------|-------------|-------------------|
| `thesis_main` | zero_shot, finetune, ANIL, FOMAML, MAML | Logistics, Packaging Material | RQ1 (method comparison), RQ7 (dept generalization), RQ8 (collapse) |
| `init_ablation` | zero_shot, finetune, ANIL, FOMAML, MAML | Logistics, Packaging Material | RQ4 (initialization quality) |
| `k_shot_label_budget` | zero_shot, finetune, ANIL, FOMAML, MAML | Logistics, Packaging Material | RQ5 (k-shot feasibility) |

`thesis_main` is the **primary experiment** producing Tables 1 and 2 and Figures 3, 11, 12, 13 of the thesis.

### Group 3 — MAML-Focused Optimization

| Preset | Methods | Departments | Notes |
|--------|---------|-------------|-------|
| `maml_lr_sweep` | ANIL, FOMAML, MAML | Logistics, Packaging Material | Inner-LR × inner-step sweep for RQ2 and RQ3 |
| `maml_freeze_ablation` | ANIL, FOMAML, MAML | Logistics | **TODO — not yet runnable.** Requires freeze-scope support in `_run_meta_method()`. Documented for planning. |
| `maml_mechanics_debug` | FOMAML, MAML | Logistics | **TODO — not yet runnable.** Same blocker as `maml_freeze_ablation`. |

### Group 4 — Stage 2b Synthetic Support Augmentation

| Preset | Methods | Departments | Research Questions |
|--------|---------|-------------|-------------------|
| `stage2b_smotenc` | ANIL | Logistics, Packaging Material | RQ6 (SMOTENC support augmentation) |
| `stage2b_label_budget` | finetune, ANIL, FOMAML, MAML | Logistics, Packaging Material | RQ6 with k-shot label budget sweep |

---

## 2. Runnable vs. TODO Presets

| Preset | Status | Blocker |
|--------|--------|---------|
| `quick_debug` | ✅ Runnable | — |
| `stage2b_debug` | ✅ Runnable | — |
| `thesis_main` | ✅ Runnable | — |
| `init_ablation` | ✅ Runnable | — |
| `k_shot_label_budget` | ✅ Runnable | — |
| `maml_lr_sweep` | ✅ Runnable | — |
| `stage2b_smotenc` | ✅ Runnable | — |
| `stage2b_label_budget` | ✅ Runnable | — |
| `maml_freeze_ablation` | ⚠ TODO | Freeze scope not implemented in `_run_meta_method()` |
| `maml_mechanics_debug` | ⚠ TODO | Same as above |

---

## 3. Stage 1 Initialization Conditions (A/B/C/D)

Stage 2 meta-learning starts from a pre-trained MLP produced by Stage 1. The four conditions are:

| Condition | Name | Stage 1 Training Target | Notes |
|-----------|------|------------------------|-------|
| A | `A_weak_only` | Snorkel probabilistic labels (`renegotiation_prob`) | **Canonical** — used in `thesis_main` |
| B | `B_gold_only` | Gold labels only | Ablation — strong signal but small dataset |
| C | `C_hybrid` | Weighted combination (gold + weak) | Ablation — full information |
| D | `D_hybrid_unweighted` | Unweighted combination | Ablation — control for weighting scheme |
| — | `random` | No pre-training (random initialization) | Lower bound — used in `init_ablation` |

The `init_ablation` preset sweeps all five conditions with ANIL as the meta-method to isolate the initialization effect (RQ4).

---

## 4. Stage 2b Invariants

SMOTENC support augmentation enforces these invariants in every experiment:

- SMOTENC is applied **only to meta-test support sets** — never to meta-training episodes, never to query sets
- `n_query_synthetic_rows == 0` in every episode (enforced and validated)
- When real support rows already exceed `target_effective_k`, SMOTENC skips and writes `gen_skip_reason = "real_support_exceeds_target_effective_k"` — this is **not a failure**
- `support_accounting.csv` and `stage2b_config.json` are written by `save_stage2_result()` for every Stage 2b experiment
- Stale folders (pre-accounting-fix, lacking `stage2b_config.json`) are excluded from accounting validation

---

## 5. Notebook Roles

| Notebook | Role | Key sections |
|----------|------|--------------|
| `17_stage2_meta_learning.ipynb` | **Experiment controller** — select a preset and run it | 10 sections mirroring YAML groups (Sections 3-8: per-group controls + runner; Sections 9-10: post-run inspection) |
| `17.1_stage2_analysis_dashboard.ipynb` | **Analysis** — answer research questions with thesis plots and tables | 10 RQ sections (RQ1-RQ10) + Supplementary artifact inspection |
| `18_dataset_diagnostic_report.ipynb` | **Diagnostics only** — 9 validity checks (no experiment execution) | Checks 2-9: dataset, gold labels, weak labels, features, task validity, Stage 1 artifacts, Stage 2 config, Stage 2b accounting |

### NB17 Section Map

| Section | Content |
|---------|---------|
| 1. Setup and Config Loading | Imports, paths, Stage 1 artifact inventory |
| 2. Available Experiment Presets | Reference table of all 10 canonical presets with CLI |
| 3. Debug / Smoke Tests | Controls for `quick_debug` and `stage2b_debug` |
| 4. Main Thesis Experiments | Controls for `thesis_main` |
| 5. Initialization Ablation | Controls for `init_ablation` |
| 6. MAML-Focused Optimization | Controls for `maml_lr_sweep` (and TODO note for freeze presets) |
| 7. K-Shot Label Budget | Controls for `k_shot_label_budget` |
| 8. Stage 2b SMOTENC | Controls for `stage2b_smotenc` and `stage2b_label_budget` with accounting check |
| 9. Post-Run Quick Summaries | Backfill, registry load, quick plot, collapse check |
| 10. Artifact/Accounting Checks | Scans all experiment folders for completeness |

### NB17.1 Section Map (Research Questions)

| Section | Research Question | Category | Key Outputs |
|---------|------------------|----------|-------------|
| RQ1 | Does meta-learning outperform zero-shot and fine-tuning? | A: Method comparison | Table 1, Figure 3, Table 2 (Stage1 vs Stage2) |
| RQ2+3 | Which MAML variant and hyperparameters work best? | B: MAML family | Tables 3, 4; Figures 4, 5 |
| RQ4 | Does initialization quality affect Stage 2 performance? | C: Initialization | Table 5, Figures 6, 7 |
| RQ5 | How does k-shot support size affect performance? | D: K-shot | Figure 8, Table 6 |
| RQ6 | Does SMOTENC support augmentation improve few-shot AUROC? | F: Stage 2b | Figures 9, 10; Stage 2b accounting |
| RQ7 | Is meta-learning generalizable across departments? | Department comparison | Figure 11 |
| RQ8 | Does meta-learning reduce prediction collapse? | G: Collapse/stability | Figure 12 |
| RQ9 | What is the adaptation trajectory across inner steps? | E: Adaptation curve | Figure 13 |
| RQ10 | Best configurations summary | Leaderboard | Table 7, Thesis summary table |
| Supp. | Per-experiment artifact inspection | — | Load any experiment by ID |

---

## 6. Plot and Table Inventory

### Tables

| ID | Title | File | Source preset |
|----|-------|------|---------------|
| Table 1 | Main Benchmark Results (best config per method) | `table_1_main_benchmark.csv` | `thesis_main` |
| Table 2 | Stage 1 vs. Stage 2 Comparison | `table_2_stage1_vs_stage2.csv` | `thesis_main` |
| Table 3 | Inner-Step Sensitivity | `table_3_step_sensitivity.csv` | `thesis_main` / `maml_lr_sweep` |
| Table 4 | LR Sensitivity and Collapse Rates | `table_4_lr_sensitivity.csv` | `maml_lr_sweep` |
| Table 5 | Initialization Ablation (ANIL, k=2) | `table_5_init_ablation.csv` | `init_ablation` |
| Table 6 | K-Shot Support Size Sensitivity | `table_6_kshot_sensitivity.csv` | `k_shot_label_budget` |
| Table 7 | Best-Run Leaderboard | `table_7_leaderboard.csv` | All presets |
| — | Thesis Summary Table (LaTeX-ready) | `thesis_summary_table.csv` | `thesis_main` |

Tables are saved to `reports/stage2/` by NB17.1.

### Figures

| ID | Title | File | Plot function | Source preset |
|----|-------|------|---------------|---------------|
| Figure 3 | Method Comparison Bar Chart | `fig3_method_comparison.pdf` | `plot_method_comparison()` | `thesis_main` |
| Figure 4 | LR Sweep: AUROC vs. Inner LR | `fig4_lr_sweep.pdf` | `plot_maml_family_comparison()` | `maml_lr_sweep` |
| Figure 5 | Collapse Rate Heatmaps | `fig5_collapse_heatmap.pdf` | `plot_collapse_rate_heatmap()` | `maml_lr_sweep` |
| Figure 6 | Initialization Comparison (ANIL) | `fig6_init_comparison.pdf` | `plot_initialization_comparison()` | `init_ablation` |
| Figure 7 | Initialization × Method Matrix | `fig7_init_method_matrix.pdf` | `plot_init_method_matrix()` | `init_ablation` |
| Figure 8 | K-Shot Sensitivity Curve | `fig8_kshot_curve.pdf` | `plot_k_shot_curve()` | `k_shot_label_budget` |
| Figure 9 | SMOTENC vs. No-Augmentation AUROC | `fig9_stage2b_comparison.pdf` | `plot_stage2b_performance_comparison()` | `stage2b_smotenc` |
| Figure 10a | Synthetic Support Counts | `fig10a_synthetic_support_counts.pdf` | `plot_synthetic_support_counts()` | `stage2b_smotenc` |
| Figure 10b | Skip Reasons | `fig10b_skip_reasons.pdf` | `plot_skip_reasons()` | `stage2b_smotenc` |
| Figure 11 | Method Comparison by Department | `fig11_department_comparison.pdf` | custom bar per dept | `thesis_main` |
| Figure 12 | Prediction Collapse by Method | `fig12_prediction_collapse.pdf` | `plot_prediction_collapse()` | `thesis_main` |
| Figure 13 | Inner-Step Adaptation Curve | `fig13_adaptation_curve.pdf` | custom line plot | `thesis_main` / `maml_lr_sweep` |

All figures are saved to `reports/stage2/` as PDF. Raw data saved as CSV alongside.

---

## 7. CLI and Notebook Usage Guide

### Running an experiment

```bash
# Smoke test (always run first after code changes)
PYTHONPATH=src .venv/bin/python experiments/run_stage2.py --preset quick_debug

# Primary thesis benchmark
PYTHONPATH=src .venv/bin/python experiments/run_stage2.py --preset thesis_main

# Stage 2b SMOTENC augmentation
PYTHONPATH=src .venv/bin/python experiments/run_stage2.py --preset stage2b_smotenc

# Force rerun (overwrite existing artifacts)
PYTHONPATH=src .venv/bin/python experiments/run_stage2.py --preset thesis_main --force-rerun

# Preview resolved grid without executing
PYTHONPATH=src .venv/bin/python experiments/run_stage2.py --preset thesis_main --preview-only
```

### Validating the pipeline

```bash
# Validate YAML config structure (11 checks)
python experiments/validate_stage2_config.py

# Validate Stage 2b accounting (stale-aware)
python experiments/validate_stage2b_accounting.py
```

### Notebook workflow

1. **NB18** — run validity checks before any experiments
2. **NB17** — select a preset (e.g. `thesis_main`), set `RUN_EXPERIMENTS = True`, run cells in the corresponding section
3. **NB17.1** — run Setup section, then jump to the relevant RQ section to generate plots and tables

---

## 8. Stage 1 vs. Stage 2 Logic

**Stage 1** pre-trains a shared MLP (256 → 128 → 1, dropout=0.1) on the full dataset using one of the four initialization conditions (A/B/C/D). The MLP learns a general renegotiation risk embedding.

**Stage 2** takes the Stage 1 pre-trained MLP as the initialization and runs meta-learning (ANIL/FOMAML/MAML) or fine-tuning on held-out episodes. Each episode has:
- A **support set** (k_pos positive, k_neg negative examples) drawn from the target department
- A **query set** (remaining gold-labeled contracts from the same department)

The meta-learner adapts the MLP on the support set for N inner steps, then evaluates on the query set. AUROC is the primary metric.

**Zero-shot** applies the Stage 1 MLP directly to the query set without any adaptation — this is the Stage 1 floor.

---

## 9. Stage 2b Logic

Stage 2b tests whether SMOTENC synthetic support augmentation helps when the real support set is small and imbalanced.

**Pipeline:**
1. Draw a real support set (k_pos=2, k_neg=2 contracts)
2. If `augmentation_method == "smote_nc"`:  
   a. Compute per-class row counts in the real support set  
   b. If real rows exceed `target_effective_k`, skip (write `gen_skip_reason`)  
   c. Otherwise, generate synthetic rows to fill up to `target_effective_k` per class  
   d. Concatenate real + synthetic rows → augmented support set  
3. Run meta-learning on the augmented support set
4. Evaluate on the real-only query set (never augmented)
5. Write `support_accounting.csv` with per-episode counts

**Accounting invariant:** `n_query_synthetic_rows == 0` always. This is enforced in `prepare_stage2_tasks()` and verified by `validate_stage2b_accounting.py`.

---

## 10. Archived Presets

The YAML contains 27 archived presets, all with `superseded_by` and `reason` fields. These were used during development and are superseded by the canonical 10. Do not run archived presets — they may be inconsistent with the current pipeline.

To view archived presets:
```bash
python -c "import yaml; cfg = yaml.safe_load(open('experiments/stage2_config.yaml')); \
  [print(k, '->', v.get('superseded_by')) for k, v in cfg.get('archived_presets', {}).items()]"
```

---

*Report generated by the Cowork pipeline restructuring session, 2026-04-30.*
