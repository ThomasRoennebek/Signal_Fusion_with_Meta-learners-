# Stage 2 Diagnostic Overview

**Date:** 2026-04-18  
**Purpose:** Complete read-only investigation of the Stage 2 meta-learning pipeline. Covers data structure, Stage 1 artifacts, algorithm internals, experiment configuration, current results, and a diagnosis with recommended next steps.  
**Supporting CSVs:** `reports/stage2_debug/`

---

## 1. Data and Label Structure

### 1A. Dataset Identity

| Item | Value |
|---|---|
| File | `data/processed/contract_with_features_labeled_with_gold.csv` |
| Total rows | 9,201 |
| Unique contracts | 2,209 |
| Gold-labeled rows | 700 |
| Unique gold-labeled contracts | 126 |
| Departments with gold labels | 14 |
| Gold positive rows | 387 |
| Gold negative rows | 313 |
| Null gold_y rows | 8,501 |

The dataset is a longitudinal (panel) table of procurement contracts. Each contract appears in multiple rows — one per `observation_year` (2015–2025). Gold labels are binary (`gold_y` = 0 or 1) and are assigned at the contract level: all rows for a labeled contract share the same label.

Features are derived from a Stage 1 preprocessor (`preprocessor.transform()`) and accessed through `feature_names_in_`. The number of features is not available without loading the preprocessor (joblib), but the preprocessor size is ~45 KB, suggesting a moderate-dimensional feature space.

### 1B. Gold Label Distribution by Department

Source: `reports/stage2_debug/gold_label_by_department.csv`

| Department | Gold Contracts | Pos Contracts | Neg Contracts | Max Balanced k | Max k (≥1 query) |
|---|---|---|---|---|---|
| Logistics | 14 | 9 | 5 | 5 | 4 |
| Packaging Material | 11 | 6 | 5 | 5 | 4 |
| Alliance, Acquisitions & PPM CoE | 10 | 5 | 5 | 5 | 4 |
| Bioprocessing & Raw Materials | 10 | 5 | 5 | 5 | 4 |
| Devices & Needles | 10 | 5 | 5 | 5 | 4 |
| Drug Product Outsourcing | 10 | 5 | 5 | 5 | 4 |
| Drug Substance Outsourcing | 10 | 5 | 5 | 5 | 4 |
| Quality, Production Services & Supplies | 10 | 5 | 5 | 5 | 4 |
| Raw Materials & Energy | 10 | 5 | 5 | 5 | 4 |
| Strategic Sourcing US Hub | 8 | 4 | 4 | 4 | 3 |
| Global Strategic Outsourcing & Devices | 8 | 5 | 3 | 3 | 2 |
| Bioprocessing and Excipients | 7 | 4 | 3 | 3 | 2 |
| HI Warehouse Expansion Program | 6 | 4 | 2 | 2 | 1 |
| Strategy, Sourcing & Negotiation CoE | 2 | 1 | 1 | 1 | 0 |

**Logistics** (the Stage 2 target) has 14 contracts: 9 positive, 5 negative. At `k=2/2` (the current default), 4 contracts are used as support and 10 remain for query, giving 20 repeated query-evaluation episodes. The maximum feasible balanced k without leaving zero query contracts is k=5 (all negatives used as support), but the practical maximum with at least 1 query contract remaining is k=4.

### 1C. Meta-Training Feasibility by k

Logistics is excluded as the target department. "Strategy, Sourcing & Negotiation CoE" has only 2 contracts and is effectively excluded for k≥2.

| k (pos=neg) | Valid meta-train departments |
|---|---|
| k=1 | 12 |
| k=2 | 11 |
| k=3 | 10 |
| k=4 | 8 |
| k=5 | 2 (Devices & Needles, Packaging Material only) |

At the current `k=2/2` setting, 11 departments are available for meta-training. This is a **very small meta-training pool** for learning a generalizable initialization — standard MAML benchmarks use hundreds or thousands of tasks. The pool shrinks further to 10 at k=3 and 8 at k=4.

### 1D. Temporal Structure

Source: `reports/stage2_debug/gold_labels_by_year.csv` and `gold_labels_dept_year_contracts.csv`

Gold labels span observation years 2015–2025. The panel is growing over time: the earliest years have ~28 unique gold-labeled contracts, growing to 126 by 2025. This means contracts labeled in later years appear in more observation-year rows. Episode sampling is **contract-aware**: the same contract cannot appear in both support and query sets, regardless of how many year-rows it has.

### 1E. Data Quirks

Two flags to monitor:

1. **35 rows where `gold_department` ≠ `department`** — the Stage 2 pipeline uses `department_col: "department"` (not `gold_department`) for department routing. Rows with mismatched values may be assigned to the wrong meta-training task.

2. **Strategy, Sourcing & Negotiation CoE** has only 2 total contracts (1 pos, 1 neg). It fails even k=1 with-query validation (`max_k_with_1_query = 0`) and will always be dropped by `filter_valid_departments` for any non-zero query requirement.

---

## 2. Stage 1 Starting Point

### 2A. Available Initializations

Three Stage 1 checkpoints exist under `models/stage_1/`:

| Init name | Epochs trained | Training signal | Val loss convergence |
|---|---|---|---|
| `A_weak_only` | 14 | Snorkel weak labels only | Early stopping at epoch 14 |
| `B_gold_only` | 11 | Gold labels only | Early stopping at epoch 11 |
| `C_hybrid` | 30 | Hybrid (weak + gold) | Early stopping at epoch 30 |

The MLP architecture (from `stage2_config.yaml` `mlp_config`) used in Stage 2 is:

```
TabularMLP(
    input_dim  = (from preprocessor.feature_names_in_),
    hidden_dim_1 = 256,
    hidden_dim_2 = 128,
    dropout    = 0.3
)
```

The `TabularMLP` class (`mlp.py`) has the structure:

```
Sequential(
    Linear(input_dim → 256),
    BatchNorm1d(256),
    ReLU,
    Dropout(0.3),
    Linear(256 → 128),
    BatchNorm1d(128),
    ReLU,
    Dropout(0.3),
    Linear(128 → 1)   ← classification head
)
```

The presence of `BatchNorm1d` layers is architecturally significant: their `running_var` buffers can become near-zero after Stage 1 training on sparse or near-constant features, causing denominator collapse (1/sqrt(var + eps) → 1/0.003 ≈ 300×) during meta-adaptation. This is the root cause of gradient explosions, and is explicitly addressed by `_safe_buffers()` in FOMAML and MAML.

Each `.pt` file is ~1.1 MB and each `.joblib` preprocessor is ~45 KB.

### 2B. Stage 1 Evaluation Metrics

The `tabular_results.csv` files in each init directory contain only **regression metrics against Snorkel weak labels**, not binary AUROC against gold labels:

| Metric | A_weak_only | C_hybrid |
|---|---|---|
| Mean Predictor RMSE | 0.511 | 0.512 |
| Elastic Net RMSE | 0.660 | 0.573 |
| XGBoost RMSE | 0.693 | 0.366 |

These are distillation-quality metrics and cannot be directly compared with Stage 2 gold-label AUROC. There is **no tabular Stage 1 gold-label baseline** from these files.

### 2C. Inferred Stage 1 Gold-Label Baseline (via zero-shot)

The Stage 2 pipeline provides this baseline implicitly: `zero_shot` evaluation runs the Stage 1 model directly on Logistics query episodes without adaptation. From the experiment registry:

| Init | AUROC (Logistics, k=2/2, n=20 episodes) |
|---|---|
| A_weak_only | **0.634 ± 0.199** |

This is the true Stage 1 Logistics baseline. **No adaptive method in the current experiments exceeds it.** This is the central finding of the current results.

---

## 3. Stage 2 Algorithm Overview

### 3A. Model used across all methods

All methods operate on the same `TabularMLP` architecture, initialized from a Stage 1 checkpoint (or randomly for the `random` init). The 9-layer `net` sequential is the shared backbone. Methods differ in which parameters they adapt and how.

### 3B. Zero-shot

**Source:** `meta_common.py → evaluate_zero_shot_on_target_episodes()`

**What is adapted:** Nothing. The Stage 1 weights are frozen.

**Evaluation:** For each of the 20 Logistics episodes, the query set is passed through the pretrained model using `torch.no_grad()`. Sigmoid probabilities are computed and evaluated against gold labels.

**Collapse detection:** `compute_prediction_stats()` is applied to each episode. Collapse is defined as: `pred_std < 0.01` OR `pred_mean < 0.02` OR `pred_mean > 0.98`. The zero-shot model never collapses (pred_std ≈ 0.138), confirming the Stage 1 network produces discriminative output for Logistics contracts.

### 3C. Finetune

**Source:** `stage2.py → run_finetune_baseline()`

**What is adapted:** All model parameters via Adam optimizer, applied per-episode.

**Key details:**
- The base model is loaded once with Stage 1 weights. For each evaluation episode, it is **deep-copied**, and `target_inner_steps` Adam steps are taken on the support set using the same `inner_lr`.
- **Critical design note:** `target_inner_steps` comes from `config["meta_config"]["target_inner_steps"]`, which defaults to **5** in `stage2_config.yaml`. The `inner_steps` grid parameter that varies across experiments is **not used** by finetune during evaluation — it always runs 5 steps. This explains why the finetune results are **identical for steps=1, 3, and 5** in the registry.
- There is no gradient clipping in the finetune baseline.
- The model is in `.train()` mode during the support update steps (BatchNorm updates running statistics), then switched to `.eval()` for query inference.

**Observed behavior:** `collapse_rate = 0.90` — 18 out of 20 episodes collapse (pred_std < 0.01). The Adam optimizer overfits to 4 support examples in 5 steps at lr=0.01, saturating the output. AUROC is 0.503 — essentially random.

### 3D. ANIL (Almost No Inner Loop)

**Source:** `anil.py → adapt_anil_on_episode()`, `meta_train_anil()`

**What is adapted:** Only the final linear layer (`net[-1]`): `head_weight` and `head_bias`. The feature extractor (all other layers) is **frozen** throughout inner loop adaptation.

**Inner loop mechanics:**
1. A single forward pass through the backbone (`net[:-1]`) is run with `torch.no_grad()` to extract support features. These features are computed once and reused.
2. The head parameters are adapted functionally (manual gradient descent) for `inner_steps` iterations using `BCEWithLogitsLoss`.
3. Gradients are computed through `head_weight` and `head_bias` with `create_graph=True` (for the outer MAML-style update during meta-training), but this is only the head — so the outer update only backpropagates through the head.

**Meta-training outer loop:**
- Optimizer: `Adam(model.parameters(), lr=outer_lr=0.001)` — but because only the head is adapted in the inner loop, the outer gradient signal primarily updates the head. Backbone gradients come from the outer query loss flowing through the frozen backbone.
- Batch departments sampled with `rng.choices` (with replacement) — same department can appear twice in one meta-batch.
- Gradient clipping: `clip_grad_norm_(max_norm=1.0)`.
- NaN query losses cause that episode to be skipped.

**Stabilization:** No `_safe_buffers` needed — BatchNorm buffers are never modified functionally since the backbone is not adapted.

**Observed behavior:** Stable — `collapse_rate = 0.00` for all step counts. AUROC improves with steps: 0.414 (steps=1) → 0.521 (steps=3) → 0.542 (steps=5). However, even the best ANIL config (steps=5) does not beat zero-shot (0.634). The head-only adaptation with 4 labeled support examples appears insufficient to improve over the pre-trained representation within 5 gradient steps.

### 3E. FOMAML (First-Order MAML)

**Source:** `FOMAML.py → adapt_fomaml_on_episode()`, `meta_train_fomaml()`

**What is adapted:** All model parameters — both the backbone and the head.

**Inner loop mechanics:**
1. `torch.func.functional_call(model, (params, buffers), X_support)` — the model is called functionally (no in-place weight modification).
2. `_safe_buffers(model)` is called before each forward pass: `running_var` values are clamped to `min=1e-3` to prevent BatchNorm denominator collapse.
3. Gradients are computed with `create_graph=False` (first-order approximation — Hessian terms are dropped).
4. Gradient NaN/Inf values are replaced with 0.0/±1.0 via `torch.nan_to_num`, then **clamped to ±1.0** per gradient tensor, before the parameter update `param = param - inner_lr * clamped_grad`.

**Meta-training outer loop:**
- Departments sampled **without replacement** when `meta_batch_size ≤ n_valid_departments` — prevents the same department appearing twice in a meta-batch.
- Outer loss = mean of valid query losses; backward propagates through the adapted params.
- Gradient clipping: `clip_grad_norm_(max_norm=1.0)`.

**Observed behavior:** Fully collapses — `collapse_rate = 1.00` for all step counts at `inner_lr=0.01`. The pred_std is 0.002–0.003 across all FOMAML experiments (far below the 0.01 threshold). The AUROC values (0.478–0.503) are misleading because they are computed on the 0% non-collapsed episodes (with 100% collapse rate, the 20/20 episodes all collapse, but the metrics function still runs and produces near-random rankings from constant predictions). FOMAML at lr=0.01 is effectively broken.

### 3F. MAML (Full Second-Order MAML)

**Source:** `maml.py → adapt_maml_on_episode()`, `meta_train_maml()`

**What is adapted:** All model parameters, same as FOMAML.

**Difference from FOMAML:** In meta-training, `create_graph=True` in `adapt_maml_on_episode()`, meaning the gradient computation graph is retained through the inner loop so the outer optimizer can compute second-order derivatives. During evaluation (`evaluate_maml_on_target_episodes`), `create_graph=False` is used.

All other stabilization logic (`_safe_buffers`, per-element grad clamp ±1.0, `nan_to_num`, `clip_grad_norm_`) is identical to FOMAML.

**Observed behavior:** Partially collapses. At `inner_lr=0.01`:
- steps=1: `collapse_rate=0.80`, AUROC=0.468
- steps=3: `collapse_rate=0.35`, AUROC=0.411
- steps=5: `collapse_rate=0.95`, AUROC=0.544

The second-order gradients appear to provide some regularization versus FOMAML at steps=3 (collapse_rate 0.35 vs 1.0), but collapse dominates at steps=5. The AUROC=0.544 at steps=5 is the best adaptive result in the registry, but with 19/20 episodes collapsed it is not a reliable estimate.

---

## 4. Experiment Configuration

### 4A. Base Configuration

The base config (`stage2_config.yaml → base_config`) defines defaults that all presets inherit unless explicitly overridden:

| Parameter | Value |
|---|---|
| Seed | 42 |
| Target column | `gold_y` |
| Department column | `department` |
| Group column (contract) | `contract_id` |
| Init name | `A_weak_only` |
| Target department | `Logistics` |
| `n_support_pos` | 2 |
| `n_support_neg` | 2 |
| `meta_batch_size` | **2** (overridden to 4 in all thesis presets) |
| `meta_iterations` | 200 |
| `inner_lr` | **0.01** |
| `outer_lr` | 0.001 |
| `inner_steps` | 3 |
| `target_inner_steps` | 5 |
| `n_repeats` | 20 |
| `ranking_k` | 10 |

### 4B. Experiment ID Scheme

Experiment IDs are generated by `generate_experiment_id()`:

```
stage2__init-{init_name}__method-{method}__kpos-{k}__kneg-{k}__steps-{steps}__target-{dept}
```

When `inner_lr_grid` is present in a preset, the LR is appended:

```
stage2__init-A_weak_only__method-fomaml__kpos-2__kneg-2__steps-3__target-Logistics__lr-0.001
```

This ensures LR-swept experiments never overwrite the `inner_lr=0.01` baseline runs. The 15 existing experiments all use the no-`__lr-` format (legacy runs at lr=0.01).

### 4C. Existing Results Coverage

All 15 experiments in `models/stage_2/experiments/experiment_summary.csv` share:

| Dimension | Value |
|---|---|
| Target department | Logistics |
| Init name | A_weak_only |
| k (pos/neg) | 2 / 2 |
| inner_lr | 0.01 |
| meta_batch_size | 4 |
| n_repeats | 20 |
| Methods | zero_shot × 3, finetune × 3, anil × 3, fomaml × 3, maml × 3 |
| Steps varied | 1, 3, 5 |

No experiments have been run with `inner_lr ≠ 0.01`, `init_name = C_hybrid`, `k ≠ 2/2`, or any department other than Logistics.

### 4D. Presets Ready to Run

| Preset | Purpose | Experiments |
|---|---|---|
| `optimization_cycle_ref` | Reference re-run (zero_shot, finetune) at meta_batch=4, n_repeats=20 | 6 |
| `optimization_cycle_adaptive` | LR sweep {0.001, 0.005, 0.01} × {anil, fomaml, maml} × steps {1,3,5} | 27 |
| `thesis_main` | Full k-sweep {2/2, 5/5, 10/10}, init sweep {A_weak_only, C_hybrid} | ~48 |
| `k_shot_sweep` | k from 1 to 5, all 5 methods, steps=3, meta_batch=4 | 25 |
| `five_method_benchmark` | All 5 methods, k=2/2, steps {1,3,5}, meta_batch=4 | 15 |

Run the optimization cycle with: `bash experiments/run_optimization_cycle.sh` (or `--adaptive-only`).

---

## 5. Results and Diagnostics

### 5A. Summary Table (All 15 Experiments)

Source: `reports/stage2_debug/stage2_results_summary.csv`

| Method | Steps | AUROC ± std | Collapse Rate | pred_std | pred_mean | NDCG@10 |
|---|---|---|---|---|---|---|
| **zero_shot** | 1 | 0.634 ± 0.199 | 0.00 | 0.138 | 0.132 | 0.789 |
| **zero_shot** | 3 | 0.634 ± 0.199 | 0.00 | 0.138 | 0.132 | 0.789 |
| **zero_shot** | 5 | 0.634 ± 0.199 | 0.00 | 0.138 | 0.132 | 0.789 |
| anil | 1 | 0.414 ± 0.173 | 0.00 | 0.112 | 0.500 | 0.575 |
| anil | 3 | 0.521 ± 0.202 | 0.00 | 0.098 | 0.504 | 0.643 |
| anil | 5 | 0.542 ± 0.201 | 0.00 | 0.098 | 0.504 | 0.647 |
| finetune | 1 | 0.503 ± 0.033 | 0.90 | 0.028 | 0.480 | 0.699 |
| finetune | 3 | 0.503 ± 0.033 | 0.90 | 0.028 | 0.480 | 0.699 |
| finetune | 5 | 0.503 ± 0.033 | 0.90 | 0.028 | 0.480 | 0.699 |
| fomaml | 1 | 0.478 ± 0.091 | 1.00 | 0.003 | 0.480 | 0.672 |
| fomaml | 3 | 0.490 ± 0.025 | 1.00 | 0.003 | 0.596 | 0.692 |
| fomaml | 5 | 0.503 ± 0.032 | 1.00 | 0.002 | 0.597 | 0.719 |
| maml | 1 | 0.468 ± 0.194 | 0.80 | 0.008 | 0.641 | 0.712 |
| maml | 3 | 0.411 ± 0.166 | 0.35 | 0.020 | 0.472 | 0.686 |
| maml | 5 | 0.544 ± 0.148 | 0.95 | 0.004 | 0.433 | 0.678 |

**Key observation:** All three zero_shot rows are identical because zero-shot is deterministic and step-count-independent. All three finetune rows are identical because finetune uses `target_inner_steps=5` from the base config regardless of which `inner_steps` value is in the grid (see Section 3C above — this is a known design issue).

### 5B. Collapse Analysis

Collapse is diagnosed per episode: a prediction vector is flagged as collapsed if `pred_std < 0.01` (all predictions near-constant) or `pred_mean < 0.02` or `pred_mean > 0.98` (model stuck predicting one class). `collapse_rate` is the fraction of the 20 evaluation episodes that collapse.

| Method | Best collapse rate | Mechanism |
|---|---|---|
| zero_shot | 0.00 | No adaptation — pred_std ≈ 0.138 |
| anil | 0.00 | Head-only: backbone BatchNorm never disturbed |
| finetune | 0.90 | Per-episode Adam at lr=0.01 overfits 4 examples in 5 steps |
| fomaml | 1.00 | Full-model gradient at lr=0.01 overwhelms BatchNorm stabilization |
| maml | 0.35 (steps=3) | Second-order provides some implicit regularization |

FOMAML's complete collapse at all step counts at lr=0.01 is the clearest signal that the learning rate is too high for whole-model adaptation with so few support examples.

MAML is erratic: best at steps=3 (collapse_rate=0.35) but degrades sharply at steps=5 (0.95), suggesting that more inner steps push the model further into an unstable region even with second-order corrections.

### 5C. Performance vs. Zero-Shot

Zero-shot (AUROC=0.634, NDCG@10=0.789) is the best performing configuration in the entire registry. Every adaptive method degrades performance relative to the Stage 1 pretrained representation:

| Method | Best AUROC | Delta vs zero-shot |
|---|---|---|
| maml (steps=5) | 0.544 | −0.090 |
| anil (steps=5) | 0.542 | −0.092 |
| finetune | 0.503 | −0.131 |
| fomaml (steps=5) | 0.503 | −0.131 |
| maml (steps=3) | 0.411 | −0.223 |

This pattern indicates that **Stage 2 adaptation, as currently configured, is actively hurting performance**. This is not unusual in small-data meta-learning regimes (the few-shot problem is genuinely hard), but it sets a clear baseline: the optimization cycle must show AUROC > 0.634 before any adaptive method can be claimed to add value.

### 5D. Log-Loss Anomaly in Finetune

Finetune shows a gold log-loss of **8.11** — extremely high, more than 3× any other method. This is the characteristic signature of a model that occasionally produces probabilities close to 0 or 1 on the wrong class: the log-loss penalty for a confident wrong prediction is unbounded. The `pred_mean=0.480` and `frac_near_zero=0.506` values confirm the model is mostly predicting near-zero probabilities. The `frac_near_one=0.446` indicates that ~45% of predictions are > 0.95 in the 10% of non-collapsed episodes, creating large log-loss contributions. Any reporting of finetune log-loss should include a caveat about the collapse rate.

---

## 6. Diagnosis Summary

### 6A. Critical Bottlenecks

**Bottleneck 1: inner_lr = 0.01 causes gradient explosions in FOMAML and MAML**

The `_safe_buffers` mechanism (clamping BatchNorm `running_var ≥ 1e-3`) and the per-element gradient clamping to `±1.0` in the inner loop are insufficient at `inner_lr=0.01` with a full-model functional update. Even with per-element clipping, the accumulation of 256-wide and 128-wide parameter vectors leads to effective parameter steps that shift the model out of the stable initialization basin. The result is complete output collapse (pred_std < 0.003) for FOMAML and near-complete collapse for MAML.

**Bottleneck 2: finetune ignores inner_steps during evaluation**

`run_finetune_baseline()` uses `config["meta_config"]["target_inner_steps"]` (hardcoded to 5 in the base config) rather than `inner_steps` from the experiment grid. All three finetune experiments in the registry are therefore duplicates. This wastes experiment budget and obscures the true step-sensitivity of finetune. The fix is either to align `target_inner_steps` with `inner_steps` in the grid, or to explicitly sweep `target_inner_steps` as a separate parameter.

**Bottleneck 3: meta-training pool is tiny (11 departments at k=2/2)**

MAML and FOMAML are designed for settings with many diverse tasks. With 11 meta-training tasks and 200 iterations, each episode is seen many times. The model is effectively overfitting to the meta-training distribution rather than learning a general initialization. This is a fundamental data regime constraint, not a code bug. It is likely a significant reason why adaptive methods cannot beat zero-shot.

**Bottleneck 4: ANIL head-only adaptation is under-powered**

With 4 support examples (2 pos, 2 neg) and only the final `Linear(128→1)` layer being adapted, the information available to the head is whatever the frozen backbone already encodes. If the Stage 1 backbone does not produce linearly separable representations for Logistics (a department not seen during meta-training), no amount of head adaptation will help. Steps=5 gives AUROC=0.542, which is less than 0.634 from zero-shot. The head update cannot overcome the backbone's distributional shift.

**Bottleneck 5: no experiments with init=C_hybrid or k≠2/2**

All 15 experiments use `A_weak_only` and `k=2/2`. It is unknown whether the C_hybrid initialization (trained on gold labels for 30 epochs) would provide a better starting representation. The ANIL and zero-shot comparison would be most informative first, since they are the only non-collapsed methods.

### 6B. Highest-Value Next Run

**Run: `optimization_cycle` — LR sweep for FOMAML and MAML**

```bash
# From project root with .venv active:
bash experiments/run_optimization_cycle.sh
```

This runs 33 experiments:
- 6 reference (zero_shot + finetune at steps 1, 3, 5) — clean re-run for comparison
- 27 adaptive (ANIL + FOMAML + MAML × steps {1,3,5} × lr {0.001, 0.005, 0.01})

**Expected outcomes:**
- At `inner_lr=0.001`: FOMAML and MAML should stabilize (`collapse_rate → 0`). If they stabilize but AUROC remains below 0.634, the problem is representation capacity, not optimization instability.
- At `inner_lr=0.005`: intermediate — likely partial stabilization. This is the most informative LR for discriminating between optimization and representation bottlenecks.
- At `inner_lr=0.01`: should reproduce current results (confirmation run).
- ANIL is included for completeness; its stability should be LR-independent, but the head update quality may still vary.

**Why this is highest-value:** It directly tests the hypothesis that LR is the primary cause of collapse. If FOMAML/MAML stabilize at lr=0.001 and beat zero-shot, the Stage 2 approach is viable at lower LR. If they stabilize but remain below zero-shot, the next bottleneck to address is representation quality — either by switching to `C_hybrid` init or by increasing meta-training task diversity.

**After the optimization cycle, the next decision point:**
- If best adaptive AUROC > 0.634 → proceed to k-shot sweep to characterize data efficiency
- If best adaptive AUROC ≤ 0.634 → test `C_hybrid` init with the best stable LR from the cycle

---

## Appendix: Supporting Files

| File | Contents |
|---|---|
| `reports/stage2_debug/gold_label_by_department.csv` | Per-department: gold rows, contracts, pos/neg contracts, max k |
| `reports/stage2_debug/gold_labels_by_year.csv` | Per-year: total rows, unique contracts, pos/neg contracts |
| `reports/stage2_debug/gold_labels_dept_year_contracts.csv` | Pivot: unique contracts by department × observation year |
| `reports/stage2_debug/stage2_results_summary.csv` | Cleaned 15-experiment results table with collapse diagnostics |
| `models/stage_2/experiments/experiment_summary.csv` | Full raw results registry (all experiments) |
| `experiments/stage2_config.yaml` | All preset definitions |
| `experiments/run_optimization_cycle.sh` | Script to run the 33-experiment LR sweep |
