# Stage 2 — Optimization Benchmark: LR Sweep Analysis

**Date:** 2026-04-18  
**Init:** A_weak_only | **Target:** Logistics | **Support:** k=2/2  
**Sweep:** inner_lr ∈ {0.001, 0.005, 0.01} × inner_steps ∈ {1, 3, 5}  
**Methods:** zero_shot (reference), ANIL, FOMAML, MAML  
**meta_batch_size:** 4 (upgraded from 2 in prior runs)

---

## 1. Existing Baseline Results (lr = 0.01, meta_batch_size = 2)

All 15 experiments (5 methods × 3 step counts) completed successfully.  
20 evaluation episodes per experiment; metrics are mean ± std across episodes.

| Method | Steps | AUROC | Log-Loss | ECE | NDCG@10 | P@10 | Collapse |
|--------|------:|------:|---------:|----:|--------:|-----:|---------:|
| **zero_shot** | — | **0.634** ± 0.199 | 2.060 | 0.585 | **0.789** | **0.790** | 0.00 |
| ANIL | 1 | 0.414 ± 0.173 | 0.881 | 0.347 | 0.575 | 0.565 | 0.00 |
| ANIL | 3 | 0.521 ± 0.202 | 0.857 | 0.337 | 0.643 | 0.635 | 0.00 |
| ANIL | 5 | 0.542 ± 0.201 | 0.855 | 0.334 | 0.648 | 0.635 | 0.00 |
| FOMAML | 1 | 0.478 ± 0.091 | 1.169 | 0.219 | 0.672 | 0.655 | **1.00** |
| FOMAML | 3 | 0.490 ± 0.025 | 0.644 | 0.123 | 0.692 | 0.690 | **1.00** |
| FOMAML | 5 | 0.503 ± 0.032 | **0.639** | **0.115** | 0.719 | 0.715 | **1.00** |
| MAML | 1 | 0.468 ± 0.194 | 2.412 | 0.350 | 0.712 | 0.685 | 0.80 |
| MAML | 3 | 0.411 ± 0.166 | 1.405 | 0.320 | 0.687 | 0.625 | 0.35 |
| MAML | 5 | 0.544 ± 0.148 | 1.368 | 0.312 | 0.678 | 0.690 | 0.95 |
| finetune | 1-5 | 0.503 ± 0.033 | 8.112 | 0.467 | 0.699 | 0.675 | 0.90 |

> Note: zero_shot results are identical for all inner_steps since no adaptation is applied.

### Key Observations from Baseline

**Zero-shot leads in AUROC and NDCG@10.** The pre-trained Stage 1 model achieves 0.634 AUROC and 0.789 NDCG@10 without any domain adaptation. This sets the bar every adaptive method must beat.

**ANIL is the only stable adaptive method.** Results improve monotonically with step count (0.41 → 0.52 → 0.54 AUROC). No collapse (pred_std ≈ 0.10 across all step counts). However, it has not yet closed the gap with zero-shot at lr=0.01.

**FOMAML suffers complete prediction collapse.** collapse_rate = 1.0 at all step counts means all 20 episodes produce nearly constant probability outputs (pred_std ≈ 0.002). Despite this, NDCG@10 is competitive (0.719 at steps=5) because even a near-constant output can maintain relative ranking if the constant hovers near the class-positive proportion. ECE is excellent (0.115), but this is partly an artefact of collapse onto the base-rate.

**MAML is unstable.** Non-monotone AUROC (0.47 → 0.41 → 0.54) with high variance. Collapse rate varies erratically. The second-order gradients are amplified by the small 2-episode meta-batch, causing noisy outer updates.

**Finetune is catastrophic.** Log-loss=8.11 indicates diverged parameters; the model predicts near-zero probability for all contracts regardless of support, causing the constant-output collapse.

---

## 2. Diagnosis: Why Does FOMAML/MAML Collapse?

Collapse means every query prediction in an episode converges to the same value, so the model provides no ranking signal. Two mechanisms cause this:

### 2a. Inner LR Too High (Most Likely Root Cause)

With `inner_lr = 0.01` and the full backbone being updated during inner-loop adaptation:

1. The support set is tiny (2 pos + 2 neg = 4 samples).
2. After 1-3 gradient steps at lr=0.01, the model can easily reach a degenerate minimum where all logits → 0 (sigmoid → 0.5) or all logits → ±∞.
3. When all logits collapse to a constant, BCEWithLogitsLoss gradient is near-zero, and the meta-learner receives no signal for the outer update either.

**Expected effect of reducing lr:**
- `lr = 0.001`: Very conservative adaptation. The model stays close to the pre-trained weights. Collapse rate should drop dramatically. Risk: insufficient adaptation signal.
- `lr = 0.005`: Moderate adaptation. Should reduce collapse while still allowing meaningful head updates.
- `lr = 0.01` (current): Too aggressive for full-backbone updates on 4 samples.

### 2b. Meta-Batch Size = 2 (Addressed in New Benchmark)

With only 2 departments per outer step, outer gradient estimates are very noisy. Meta-batch=4 provides a more stable MAML outer update and is used in the new benchmark.

### 2c. BatchNorm Running Statistics (Fixed)

The `_safe_buffers()` clamp on `running_var ≥ 1e-3` prevents the 300× amplification that caused NaN gradients. This fix is in place.

---

## 3. Framework Changes Implemented

The following code changes have been made to support the new benchmark:

### `src/master_thesis/stage2.py`

**`generate_experiment_id`** now accepts an optional `inner_lr` parameter:
```python
def generate_experiment_id(..., inner_lr: Optional[float] = None) -> str:
    id_str = f"stage2__init-{init_name}__method-{method}__kpos-{n_support_pos}..."
    if inner_lr is not None:
        lr_str = f"{inner_lr:.6f}".rstrip("0").rstrip(".")
        id_str += f"__lr-{lr_str}"
    return id_str
```
When `inner_lr_grid` is in the preset, each experiment gets a unique ID like  
`stage2__init-A_weak_only__method-fomaml__kpos-2__kneg-2__steps-3__target-Logistics__lr-0.005`.  
This prevents lr variants from overwriting each other.

**`resolve_experiment_grid`** now handles `inner_lr_grid`:
```python
inner_lr_grid_raw = experiment_grid.get("inner_lr_grid", None)
sweep_lr = inner_lr_grid_raw is not None
inner_lr_grid = [float(lr) for lr in inner_lr_grid_raw] if sweep_lr else [default_lr]

for method, support_cfg, inner_steps, init_name, target_department, inner_lr in product(...):
    resolved["meta_config"]["inner_lr"] = inner_lr
    exp_id = generate_experiment_id(..., inner_lr=inner_lr if sweep_lr else None)
```

### `experiments/stage2_config.yaml`

New preset `lr_sweep_benchmark`:
```yaml
lr_sweep_benchmark:
  methods: ["anil", "fomaml", "maml"]
  support_grid:
    - n_support_pos: 2
      n_support_neg: 2
  inner_steps_grid: [1, 3, 5]
  inner_lr_grid: [0.001, 0.005, 0.01]
  init_names: ["A_weak_only"]
  meta_batch_size: 4
  n_repeats: 20
```

**Total experiments:** 3 methods × 3 steps × 3 lr = **27 experiments**

---

## 4. Running the New Benchmark

The benchmark cannot execute in the code-analysis sandbox (no torch/sklearn in the Linux environment). It must be run on the project's Mac environment with the venv.

### Option A — Shell script (recommended)

```bash
cd "path/to/Master Thesis"
source .venv/bin/activate
bash experiments/run_lr_sweep.sh
# Or to force-rerun existing results:
bash experiments/run_lr_sweep.sh --force-rerun
```

### Option B — Jupyter notebook

Open `notebooks/18.1_lr_sweep_benchmark.ipynb` in JupyterLab and run all cells.  
The notebook runs all 27 experiments, displays heatmaps, and saves `reports/lr_sweep_results.csv`.

### Option C — Python one-liner

```bash
cd "path/to/Master Thesis"
source .venv/bin/activate
PYTHONPATH=src python experiments/run_stage2.py \
  --config experiments/stage2_config.yaml \
  --preset lr_sweep_benchmark
```

---

## 5. What to Expect from the LR Sweep

Based on standard meta-learning theory and the collapse diagnosis above:

### FOMAML — Expected behaviour across lr values

| lr | Expected collapse_rate | Expected AUROC | Notes |
|----|----------------------:|---------------:|-------|
| 0.001 | 0.0 – 0.2 | 0.55 – 0.65 | Near pre-trained behaviour; good calibration |
| 0.005 | 0.1 – 0.5 | 0.50 – 0.62 | Balanced adaptation |
| 0.01 | **1.00** (observed) | ~0.50 | Full collapse confirmed |

At lower lr, FOMAML should produce spread predictions (pred_std will rise from 0.002 toward 0.05+), enabling real discrimination.

### MAML — Expected behaviour across lr values

| lr | Expected collapse_rate | Expected AUROC | Notes |
|----|----------------------:|---------------:|-------|
| 0.001 | 0.0 – 0.3 | 0.52 – 0.65 | Second-order correction stable at small steps |
| 0.005 | 0.2 – 0.6 | 0.48 – 0.60 | Partial collapse |
| 0.01 | 0.35–0.95 (observed) | ~0.48 (unstable) | Confirmed unstable |

MAML with `meta_batch_size=4` should also show improved stability over the prior `meta_batch_size=2` runs.

### ANIL — Expected behaviour across lr values

ANIL adapts only the classification head (final linear layer), not the full backbone.  
The head has far fewer parameters (hidden_dim_2 × 1 = 128 parameters) than the full backbone (~100k).  
This makes ANIL inherently more stable:

| lr | Expected impact |
|----|----------------|
| 0.001 | Very conservative head updates; close to zero-shot |
| 0.005 | Moderate improvement over steps=5 baseline |
| 0.01 (current best) | AUROC=0.542 at steps=5; likely near optimal for ANIL |

ANIL is unlikely to benefit much from lower lr, but the sweep confirms this.

### Target: Beat Zero-Shot (AUROC > 0.634)

For any adaptive method to beat zero-shot at k=2/2:
- It must learn a Logistics-specific decision boundary from 4 support samples
- The prior representation (Stage 1 weak supervision) must be close enough that 1–5 gradient steps can refine it without destroying what Stage 1 learned

The most likely winner is **ANIL at lr=0.01, steps=5** (already at 0.542) or **FOMAML at lr=0.001, steps=5** if collapse is resolved. If the gap remains, the **k-shot sweep** (more support samples) is the next lever.

---

## 6. Interpreting Results After Running

Once the benchmark completes, compare against zero-shot by looking at:

1. **Collapse resolved?** — Check `collapse_rate` for FOMAML and MAML at lr=0.001. If it drops from 1.0 toward 0.0, the hypothesis is confirmed.

2. **AUROC improvement** — Does any configuration exceed 0.634?

3. **Calibration** — ECE close to 0 is desirable, but only meaningful when collapse_rate=0 (otherwise the "calibration" is artefactual).

4. **NDCG@10 improvement** — The most thesis-relevant metric (ranking top contracts for manual review). If FOMAML/MAML collapse resolves, their NDCG@10 should exceed the current 0.719 and possibly beat zero-shot's 0.789.

5. **Best overall configuration** — Summarise as:  
   `{method} (lr={best_lr}, steps={best_steps})` vs zero-shot reference.

---

## 7. Summary Table Template

After running, fill in this table:

| Method | Best lr | Best steps | AUROC | Log-Loss | ECE | NDCG@10 | Collapse |
|--------|--------:|----------:|------:|---------:|----:|--------:|--------:|
| zero_shot | — | — | 0.634 | 2.060 | 0.585 | 0.789 | 0.00 |
| ANIL | ? | ? | ? | ? | ? | ? | 0.00 |
| FOMAML | ? | ? | ? | ? | ? | ? | ? |
| MAML | ? | ? | ? | ? | ? | ? | ? |

---

*Report generated by Claude — Stage 2 Optimization Benchmark framework, April 2026.*
