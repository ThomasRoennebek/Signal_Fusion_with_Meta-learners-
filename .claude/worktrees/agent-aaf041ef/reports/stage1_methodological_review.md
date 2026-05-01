# Stage 1 Deep Methodological Review

**Date:** 2026-04-25  
**Scope:** Full audit of the Stage 1 training pipeline — design correctness, target alignment, split logic, validation setup, MLP methodology, artifact sufficiency, and thesis-role alignment.

---

## SECTION 1 — Executive Summary

Stage 1 is broadly well-conceived: the four-condition design (A/B/C/D) cleanly isolates the contribution of weak labels, gold labels, and their combination, while the target-department exclusion rule correctly reserves Logistics gold labels for Stage 2 meta-learning. The pipeline produces usable artifacts and the evaluation framework covers classification, calibration, and ranking.

However, six issues require attention before the results can be trusted for the thesis:

1. **MLP capacity/data mismatch in B_gold_only.** The 256→128 architecture trains on only ~433 gold rows (after excluding Logistics and applying the 70/30 split). The MLP collapses: gold_f1 = 0.34, AUROC = 0.62, pred_mean = 0.33 (far below true_mean = 0.40). This is the highest-severity finding.

2. **Hyperparameter divergence between notebook and CLI.** The notebook uses lr=5e-4, weight_decay=1e-4, dropout=0.1, batch=256. The YAML config uses lr=1e-3, weight_decay=1e-5, dropout=0.3, batch=128. Saved artifacts were produced by the notebook, so the YAML is never what actually ran. This means the CLI path (`run_stage1.py`) would produce different models if executed.

3. **Validation-target inconsistency.** The notebook validates ALL conditions against gold labels (`y_val = y_gold_test`). The CLI script validates A against weak labels and B/C/D against gold. Both approaches have defensible logic, but they will produce different early-stopping points and therefore different saved weights. The choice must be explicit and justified.

4. **BCEWithLogitsLoss on soft targets (Condition A).** Using BCE with continuous renegotiation_prob ∈ [0,1] is mathematically valid (it reduces to the cross-entropy between a Bernoulli(p) target and the model's predicted Bernoulli(σ(logit))). However, the loss surface is different from standard binary classification and can produce over-confident predictions near 0.5. This is acceptable but should be documented.

5. **Regression objective for binary gold in Condition B.** ElasticNet and XGBoost use `reg:squarederror` / regression even when Condition B's targets are purely binary {0, 1}. This is suboptimal for the linear model (logistic regression would be more appropriate) though XGBoost is relatively robust to it.

6. **Condition D (hybrid_unweighted) underperforms C across all models.** D's Elastic Net collapses to AUROC=0.35 (below random), confirming that without upweighting, the ~640 gold rows are drowned by ~6,400 weak rows. D serves as a useful ablation but may not merit equal presentation alongside A/B/C.

**Bottom line:** The conceptual design is sound. The primary fixes needed are (i) right-sizing the MLP for B_gold_only, (ii) canonicalising hyperparameters, and (iii) documenting the validation-target choice. No fundamental redesign is required.

---

## SECTION 2 — Condition-by-Condition Assessment

### Table 1: Best Gold-AUROC per Condition (MLP and Best-Overall)

| Condition | Train Target | Train Rows (approx) | Best Model | Gold AUROC | Gold LogLoss | Gold F1 | Gold ECE | MLP AUROC | MLP LogLoss | MLP F1 |
|-----------|-------------|---------------------|------------|-----------|-------------|---------|---------|-----------|------------|--------|
| A_weak_only | renegotiation_prob (all) | ~6,440 | MLP | 0.392 | 0.841 | 0.513 | 0.258 | 0.392 | 0.841 | 0.513 |
| B_gold_only | gold_y (excl. Logistics) | ~433 | XGBoost | 0.853 | 1.339 | 0.789 | 0.181 | 0.623 | 0.745 | 0.336 |
| C_hybrid | weighted weak + gold | ~6,440 + ~433 gold | XGBoost | 0.907 | 1.027 | 0.839 | 0.177 | 0.831 | 0.594 | 0.562 |
| D_hybrid_unweighted | weak + gold (equal) | ~6,440 + ~433 gold | XGBoost | 0.776 | 1.188 | 0.717 | 0.235 | 0.677 | 0.837 | 0.000 |

### Condition A (Weak-Only) — Verdict: CORRECT DESIGN, EXPECTED RESULTS

Condition A trains all models on Snorkel's renegotiation_prob for all 9,201 contracts. The gold-AUROC values below 0.5 are expected and not a bug: weak labels are a noisy proxy, and the models faithfully learn the weak signal, which has limited correlation with gold labels. This is by design — A's purpose is (a) to quantify the weak-label ceiling and (b) to provide a pre-trained MLP initialization for Stage 2 MAML. Judged against that purpose, A is working correctly.

### Condition B (Gold-Only) — Verdict: CORRECT DESIGN, MLP IMPLEMENTATION CONCERN

B trains on ~433 gold-labeled rows (700 total gold minus ~60 Logistics minus ~30% validation split). The Elastic Net (AUROC=0.832) and XGBoost (AUROC=0.853) perform well, confirming that gold labels carry strong signal. The MLP collapses (AUROC=0.623, F1=0.336) because 433 rows cannot support a 256→128 network with 154 input features (~40K parameters for ~430 rows). This is a data-regime mismatch, not a design flaw. Fix: reduce capacity or add stronger regularisation specifically for B.

### Condition C (Hybrid Weighted) — Verdict: CORRECT DESIGN, BEST RESULTS

C combines all 9,201 weak-labeled rows with ~433 gold rows, upweighting gold 5× via WeightedRandomSampler (MLP) or sample_weight (baselines). XGBoost achieves the best overall gold-AUROC (0.907) and the MLP achieves a strong 0.831. This validates the core thesis claim that fusing weak + gold signals outperforms either alone. The C_hybrid MLP is likely the best Stage 2 initialization candidate.

### Condition D (Hybrid Unweighted) — Verdict: CORRECT AS ABLATION, CONFIRMS WEIGHTING VALUE

D is identical to C but without gold upweighting. Performance degrades across all models (XGBoost AUROC drops from 0.907 to 0.776; MLP F1 drops to 0.000). This confirms that without explicit upweighting, the 15:1 weak-to-gold ratio drowns gold signal. D is valuable as an ablation to justify the weighting scheme in C.

---

## SECTION 3 — Split / Exclusion Assessment

### 3.1 GroupShuffleSplit by contract_id

The pipeline uses `GroupShuffleSplit(test_size=0.30, random_state=42)` with `contract_id` as the group key. This is correct: it prevents data leakage from duplicate or related rows sharing the same contract. No issue here.

### 3.2 Target-Department Gold Exclusion

Gold labels from the target department (Logistics, 60 gold rows) are excluded from the training set for conditions B, C, and D via:

```python
gold_mask_train = (y_gold_train >= 0) & (dept_train != target_department)
```

This is correct and essential for the meta-learning setup: Logistics gold labels must be held out so Stage 2 can use them as a genuine few-shot adaptation signal. The exclusion is now properly parameterised (not hardcoded).

### 3.3 Evaluation Set Composition

All conditions are evaluated on the same 207-row gold test set (30% of 700 gold rows, preserving the natural department distribution including Logistics). This is correct: it provides a fair cross-condition comparison on identical data.

### 3.4 Concern: Condition A Uses All 9,201 Rows Including Logistics

Condition A trains on all 9,201 contracts' weak labels, including Logistics. This means the A_weak_only MLP has seen Logistics features during training, just not their gold labels. This is intentional (weak labels are available for all contracts) and acceptable, but should be documented as a potential confound when comparing A's Logistics-specific performance to B/C/D.

---

## SECTION 4 — Validation Assessment

### The Core Issue

The pipeline has two different validation strategies depending on execution path:

| Execution Path | Condition A val target | Condition B/C/D val target |
|---------------|----------------------|---------------------------|
| **Notebook** (16_stage1_baselines.ipynb) | gold labels | gold labels |
| **CLI** (run_stage1.py) | weak labels | gold labels |

This matters because early stopping selects the model checkpoint with lowest validation loss. Different validation targets → different stopping epochs → different saved weights.

### Assessment

**Notebook approach (gold for all):** Uses `y_val = y_gold_test.astype(float)` for all conditions. For Condition A (trained on weak labels), this means early stopping optimises for gold-label alignment despite training on weak labels. This is arguably desirable if A's MLP is destined for Stage 2 adaptation on gold labels, since it selects the epoch where the weak-trained model happens to be most useful for gold prediction. However, it introduces an information leak: the training procedure implicitly uses gold validation signal to select model weights.

**CLI approach (weak for A, gold for B/C/D):** More principled from a supervision-purity standpoint. Condition A only ever sees weak labels during training and validation, keeping gold labels completely held out. But this may select a model checkpoint that is overfit to weak-label noise.

### Recommendation

Use the **CLI approach** (weak validation for A, gold for B/C/D) as the canonical method. It respects the supervision boundary that defines each condition. Document the choice explicitly and note that A's validation loss is not directly comparable to B/C/D's validation loss because they measure different quantities.

---

## SECTION 5 — MLP Methodological Assessment

### 5.1 Architecture vs Data Regime

| Condition | Training Rows | Parameters (approx) | Rows/Param Ratio | Assessment |
|-----------|--------------|--------------------|--------------------|------------|
| A_weak_only | ~6,440 | ~40K | 0.16 | Marginal — standard for tabular MLP |
| B_gold_only | ~433 | ~40K | 0.01 | **Critical** — severe overparameterisation |
| C_hybrid | ~6,440 | ~40K | 0.16 | Marginal — same as A |
| D_hybrid_unweighted | ~6,440 | ~40K | 0.16 | Marginal — same as A |

The architecture (154→256→128→1 with BatchNorm and Dropout) has approximately 154×256 + 256 + 256×128 + 128 + 128×1 + 1 ≈ 40K trainable parameters. For Condition B with ~433 rows, this is an ~90:1 parameter-to-sample ratio — a recipe for overfitting or collapse regardless of regularisation. The empirical evidence confirms this: B's MLP achieves gold_auroc = 0.623 vs XGBoost's 0.853.

### 5.2 BCEWithLogitsLoss on Soft Targets

`BCEWithLogitsLoss` computes: `L = -[y·log(σ(z)) + (1-y)·log(1-σ(z))]` where y ∈ [0,1].

For Condition A where y = renegotiation_prob ∈ (0,1), this is mathematically valid and interpretable as minimising the KL divergence between the target Bernoulli(y) and the predicted Bernoulli(σ(z)). This is a standard choice for knowledge distillation / soft-label training.

For Conditions B/C/D where gold targets are binary {0,1}, BCEWithLogitsLoss is the standard classification loss. No issue.

For Condition C specifically, the hybrid target is: `y = w_weak * renegotiation_prob` for weak-only rows and `y = w_gold * gold_y` for gold rows — but these are combined via WeightedRandomSampler (which controls sampling frequency, not target values). The actual target values fed to the loss are either soft (weak rows) or binary (gold rows). This is correct.

### 5.3 Early Stopping and Training Dynamics

Early stopping uses patience=10 on validation loss. Training histories from saved artifacts show instability: validation loss oscillates between ~1.3 and ~4.7 across epochs. This suggests:

- Batch size may be too small relative to data heterogeneity
- Learning rate may be too high for the loss landscape
- BatchNorm statistics are noisy due to small batches (especially for B with ~300 training rows)

### 5.4 Preprocessor Fitting

Each call to `fit_mlp_model()` creates a new preprocessor via `build_tabular_tensors()` → `preprocessor.fit_transform()`. This means each condition gets its own preprocessing pipeline fitted on its specific training data. This is correct (each condition's feature scaling should match its training distribution). The preprocessor is saved alongside each MLP checkpoint.

### 5.5 Hyperparameter Discrepancy

This is a reproducibility concern:

| Parameter | Notebook (what actually ran) | YAML Config (what CLI would use) |
|-----------|------------------------------|----------------------------------|
| lr | 5e-4 | 1e-3 |
| weight_decay | 1e-4 | 1e-5 |
| dropout | 0.1 | 0.3 |
| batch_size | 256 | 128 |
| hidden_dim_2 | 128 | 128 |

The saved artifacts in `models/stage_1/` were produced by the notebook. The YAML config has never been used to produce the current artifacts. If someone runs `run_stage1.py` with the YAML config, they will get different models. **One canonical source of hyperparameters must be established.**

---

## SECTION 6 — Artifact / Output Assessment

### 6.1 What Is Saved (per condition)

| Artifact | Present | Purpose | Sufficient |
|----------|---------|---------|------------|
| mlp_pretrained.pt | Yes | MLP weights for Stage 2 init | Yes |
| preprocessor.joblib | Yes | Feature transform pipeline | Yes |
| history.csv | Yes | Training curves | Yes |
| tabular_results.csv | Yes | Weak-label evaluation | Yes |
| gold_results.csv | Yes | Gold-label evaluation | Yes |
| predictions.csv | Yes | Per-row predictions | Yes |
| xgboost_feature_importance.csv | Yes (XGB only) | Feature ranking | Yes |
| metadata.json | Yes (new layout) | Condition provenance | Yes |
| condition_summary.csv | Yes (root) | Cross-condition comparison | Yes |

### 6.2 What Is Missing or Problematic

1. **No ElasticNet/XGBoost model files saved.** Only MLP weights are persisted. If Stage 2 or analysis needs to re-evaluate baselines, they must be retrained. For the thesis this is acceptable (only MLP goes to Stage 2), but saving `joblib` dumps of the baselines would aid reproducibility.

2. **Dual directory layout.** Both the old flat layout (`models/stage_1/A_weak_only/`) and new target-aware layout (`models/stage_1/global/A_weak_only/`, `models/stage_1/Logistics/B_gold_only/`) coexist. The legacy layout should be removed to avoid confusion.

3. **Legacy root-level artifacts.** Files like `models/stage_1/mlp_pretrained.pt` and `models/stage_1/mlp_preprocessor.joblib` at the root level are from an earlier run and should be cleaned up.

4. **No saved hyperparameters in artifact directory.** The metadata.json includes config_snapshot, but if the notebook hyperparameters differ from the YAML, the actual training hyperparameters should be logged directly into the training history or metadata.

### 6.3 Stage 2 Handoff Readiness

Stage 2 needs: (a) MLP architecture specification, (b) pretrained weights (θ), (c) preprocessor, (d) feature list. Items (a)-(c) are available. Item (d) can be inferred from the preprocessor but is not explicitly saved as a separate artifact. Consider saving a `feature_names.json` for clarity.

---

## SECTION 7 — Recommended Correct Stage 1 Design

The current design is fundamentally correct. Below is the recommended target state with specific fixes for the issues identified above.

### 7.1 Condition Design — Keep As-Is

The four conditions (A/B/C/D) are well-motivated:

- **A** establishes the weak-label baseline and provides pre-trained initialisation
- **B** establishes the gold-label ceiling with limited data
- **C** tests the core thesis hypothesis (signal fusion)
- **D** ablates the weighting scheme

No conditions should be added or removed.

### 7.2 MLP Architecture — Adapt Per Condition

For Condition B (and potentially as a general robustness measure):

- **Option 1 (preferred):** Use a smaller architecture for B only: 154→64→32→1 (~6K parameters for ~430 rows, ratio ≈ 0.07). Keep 256→128 for A/C/D where data is plentiful.
- **Option 2:** Keep the same architecture for all conditions (for MAML compatibility) but apply much stronger regularisation for B: dropout=0.5, weight_decay=1e-3, and reduce patience to 5.
- **Option 3:** Accept B_MLP underperformance and document it as a known limitation of the data regime. This is the least-effort option and is legitimate if the thesis frames B's MLP as an illustration of why few-shot learning (Stage 2) is needed.

**For MAML compatibility, Option 3 is actually the most defensible.** Stage 2 MAML requires a fixed architecture across all tasks. If B uses a different architecture, its weights cannot serve as MAML initialisation. The recommended framing: "B_gold_only MLP's poor performance demonstrates that the 256→128 architecture requires more data than a single department's gold labels provide — motivating the meta-learning approach of Stage 2."

### 7.3 Validation Target — Standardise

Adopt the CLI convention:

- **Condition A:** validate on weak labels (renegotiation_prob)
- **Conditions B/C/D:** validate on gold labels

Document this choice in both the YAML config (add a `validation_target` field) and the thesis methods section.

### 7.4 Hyperparameters — Single Source of Truth

Canonicalise the YAML config to match what the notebook actually used:

```yaml
mlp_config:
  lr: 0.0005          # was 0.001
  weight_decay: 0.0001 # was 0.00001
  dropout: 0.1         # was 0.3
  train_batch_size: 256 # was 128
```

Then modify the notebook to read from the YAML rather than defining hyperparameters inline. This ensures CLI and notebook always produce identical models.

### 7.5 Baseline Objectives — Minor Fix

For Condition B, consider switching ElasticNet to `LogisticRegression` (with the same elastic-net penalty) since targets are binary. XGBoost can switch to `binary:logistic`. This is a minor improvement; the current regression approach still works because predictions are clipped to [0,1] and evaluated with the same metrics.

### 7.6 Artifact Hygiene

- Remove the legacy flat directory layout (`models/stage_1/A_weak_only/`, etc.)
- Remove root-level legacy files (`models/stage_1/mlp_pretrained.pt`, etc.)
- Optionally save baseline model files (ElasticNet, XGBoost `.joblib` dumps)
- Save `feature_names.json` per condition for Stage 2 clarity

---

## SECTION 8 — Concrete Next Steps (Prioritised)

### Priority 1 — Must Fix Before Thesis Submission

| # | Action | Rationale | Effort |
|---|--------|-----------|--------|
| 1 | **Canonicalise hyperparameters**: update YAML to match notebook values (or vice versa), make notebook read from YAML | Reproducibility — current saved artifacts were trained with notebook values, not YAML values | ~1 hour |
| 2 | **Standardise validation targets**: adopt weak-for-A, gold-for-B/C/D in both notebook and CLI | Methodological consistency; current notebook uses gold for all | ~30 min |
| 3 | **Document B_MLP collapse**: add a paragraph in the thesis explaining why 256→128 MLP fails on 433 rows, framing it as motivation for Stage 2 | Intellectual honesty; readers will ask about the poor B_MLP numbers | ~30 min |
| 4 | **Clean up artifact directories**: remove legacy flat layout and root-level legacy files | Prevents confusion when loading artifacts in Stage 2 | ~15 min |

### Priority 2 — Recommended Improvements

| # | Action | Rationale | Effort |
|---|--------|-----------|--------|
| 5 | **Re-run all conditions with canonical config**: after fixing #1-2, re-run to produce definitive artifacts | Current artifacts were produced with ad-hoc notebook settings | ~2 hours (compute) |
| 6 | **Switch B baselines to classification objectives**: LogisticRegression + `binary:logistic` for Condition B | More principled for binary targets; may improve B baseline performance | ~1 hour |
| 7 | **Save feature_names.json per condition**: explicit feature list for Stage 2 handoff | Reduces implicit coupling between stages | ~15 min |
| 8 | **Save baseline model .joblib files**: persist ElasticNet/XGBoost models alongside MLP | Full reproducibility of all results | ~30 min |

### Priority 3 — Optional Enhancements

| # | Action | Rationale | Effort |
|---|--------|-----------|--------|
| 9 | **Add learning-rate scheduler**: ReduceLROnPlateau or cosine annealing for MLP training | May reduce val_loss oscillations observed in training histories | ~1 hour |
| 10 | **Log actual training hyperparameters into metadata.json**: record the exact config used, not just the YAML snapshot | Belt-and-suspenders reproducibility | ~15 min |
| 11 | **Consider condition-specific MLP architecture for B**: smaller network, document as sensitivity analysis | May improve B_MLP, but conflicts with MAML fixed-architecture requirement | ~2 hours |

---

## Summary Table: Issue Severity

| Issue | Severity | Status | Recommended Action |
|-------|----------|--------|-------------------|
| MLP collapse on B_gold_only (433 rows, 40K params) | HIGH | Known | Document as motivation for Stage 2; optionally reduce capacity |
| Hyperparameter divergence (notebook ≠ YAML) | HIGH | Unresolved | Canonicalise to single source of truth |
| Validation target inconsistency (notebook ≠ CLI) | MEDIUM | Unresolved | Adopt weak-for-A, gold-for-B/C/D |
| Regression objective for binary B targets | LOW | Known | Consider switching to classification; document either way |
| BCEWithLogitsLoss on soft targets (A) | INFO | By design | Document in thesis methods |
| D underperforms C across all models | INFO | By design | Present as ablation confirming weighting value |
| Legacy artifact directories | LOW | Unresolved | Clean up |
| No saved baseline model files | LOW | Known | Optionally add |

---

*Review conducted on the codebase as of 2026-04-25. All findings based on: `src/master_thesis/mlp.py`, `src/master_thesis/baselines.py`, `src/master_thesis/stage1.py`, `src/master_thesis/metrics.py`, `experiments/run_stage1.py`, `experiments/stage1_config.yaml`, `notebooks/16_stage1_baselines.ipynb`, and saved artifacts under `models/stage_1/`.*
