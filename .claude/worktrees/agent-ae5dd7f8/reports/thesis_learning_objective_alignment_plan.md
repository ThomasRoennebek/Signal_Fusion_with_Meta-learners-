# Thesis Writing Plan: Alignment with Learning Objectives

This document gives a practical structure for writing the thesis so that each chapter clearly demonstrates compliance with the generic and specific learning objectives.

---

## 1. Overall Writing Strategy

The thesis should not be written as “MAML must win.” It should be written as an evaluated decision-support framework for contract renegotiation under label scarcity.

The central narrative should be:

> This thesis investigates whether weak supervision, multi-view signal fusion, and few-shot adaptation can support contract renegotiation prioritization in a realistic enterprise setting with scarce gold labels, heterogeneous departments, structured missingness, and leakage risk.

This framing is important because it makes the thesis defensible even if the strongest empirical result is that weak-supervised pretraining plus simple fine-tuning performs better than full MAML under limited task diversity.

---

## 2. Recommended Thesis Structure

### Chapter 1 — Introduction

Purpose: Motivate the business problem and explain why it is analytically difficult.

Include:
- Contract renegotiation as a procurement prioritization problem.
- Scarcity and uneven distribution of expert gold labels.
- Department-specific renegotiation heuristics.
- Multi-source data complexity.
- Need for interpretable decision support.

Learning objectives covered:
- Relate research to applied business context.
- Analyze decision-support needs in procurement and contract management.
- Communicate the problem to both academic and non-technical stakeholders.

Suggested wording:

> The purpose of the thesis is not to automate renegotiation decisions, but to support procurement teams by ranking contracts according to estimated renegotiation pressure and explaining the signals driving these recommendations.

---

### Chapter 2 — Literature Review

Purpose: Show that the thesis is grounded in international research.

Organize the literature review around four research streams:

1. Weak supervision and data programming  
   Discuss Snorkel, labeling functions, probabilistic labels, and label scarcity.

2. Multi-view and multi-source learning  
   Discuss heterogeneous data, view-level missingness, and signal fusion.

3. Meta-learning and few-shot adaptation  
   Discuss MAML, FOMAML, ANIL, episodic learning, support/query sets, and task diversity.

4. Explainable decision support  
   Discuss SHAP, calibration, ranking metrics, and stakeholder interpretability.

Learning objectives covered:
- Identify, understand, and critically assess international research.
- Relate ensemble learning, multi-view learning, meta-learning, and decision support to the applied business case.

Important critical angle:

> The literature often evaluates meta-learning on clean benchmark datasets with many balanced tasks. This thesis evaluates the same family of methods under realistic enterprise constraints, where tasks are few, labels are uneven, and data views are incomplete.

---

### Chapter 3 — Data Foundation and Exploratory Diagnostics

Purpose: Demonstrate that the dataset is complex, heterogeneous, and suitable for weak supervision, but challenging for standard supervised learning and full meta-learning.

Include the following figures/tables:
- Contract distribution by department.
- Gold label distribution by department.
- Gold label coverage by department.
- Stage 2 task-validity matrix for k = 2, 5, 10.
- Multi-view missingness summary.
- Snorkel probability distribution by department.
- Required downstream column validation table.

Learning objectives covered:
- Analyze complex, heterogeneous data.
- Manage and prepare multi-source datasets.
- Handle missing data and differing data coverage.
- Communicate data limitations clearly.

Key thesis message:

> The dataset contains enough unlabeled contracts to support weak supervision, but too few balanced gold labels across departments to rely on standard supervised learning or unrestricted full MAML meta-training.

Important wording:

> Label scarcity is not only a matter of total label count. The decisive constraint for meta-learning is the number of departments with both positive and negative labels available for support and query construction.

---

### Chapter 4 — Feature Engineering and Leakage Control

Purpose: Show that the pipeline is built carefully and avoids misleading signals.

Include:
- Description of the six data views:
  - contract core
  - financial
  - ESG
  - news
  - market/stock
  - macro-logistics
- Feature engineering logic for important features:
  - `lpi_below_supplier_median`
  - `years_to_expiry_capped`
  - `is_old_and_near_expiry`
  - `supplier_is_publicly_listed`
  - `esg_below_industry_min`
  - financial stress flags
- Missingness-as-signal logic.
- Correlation pruning and feature pruning audit.
- Leakage screening against labels.
- Protected feature list and downstream validation.

Learning objectives covered:
- Assess risk of information leakage.
- Evaluate time-consistent feature construction.
- Prepare large-scale multi-source data.
- Assess production readiness and replicability.

Suggested subsection:

#### 4.X Leakage and Time-Consistency Controls

Explain:
- Why multi-source joins can accidentally introduce leakage.
- Why gold labels must never be used as features.
- Why label-correlated features are audited.
- Why contract identifiers and grouping variables are protected.
- How `observation_year` supports time-aware analysis.
- How future work or extended validation can use time-based splits.

Suggested wording:

> In a multi-view enterprise dataset, leakage can arise not only from explicit target variables but also from join artifacts, duplicated identifiers, or features constructed after the decision point. The feature engineering pipeline therefore includes leakage screening, correlation-based pruning, protected-feature validation, and a saved pruning audit table.

---

### Chapter 5 — Stage 1: Weak Supervision and Global Representation Learning

Purpose: Show how weak supervision solves the first part of the label-scarcity problem.

Include:
- Labeling function design.
- LF groups:
  - lifecycle
  - financial
  - ESG
  - news
  - market
  - macro-logistics
  - department-specific rules
- LF coverage, abstain, and conflict diagnostics.
- Snorkel LabelModel output.
- Stage 1 model conditions:
  - A: weak-only
  - B: gold-only
  - C: hybrid
- Baselines:
  - mean predictor
  - Elastic Net
  - XGBoost
  - MLP

Learning objectives covered:
- Apply statistical and machine learning methods in Python.
- Critically evaluate assumptions and results.
- Develop modular multi-source modeling framework.
- Generate interpretable contract-level signals.

Important distinction:

> Regression metrics against `renegotiation_prob` measure weak-label distillation quality, not true predictive validity. Gold-label evaluation is the primary validity check.

Suggested result interpretation:

> If weak-only training performs competitively on gold labels, this supports the claim that domain heuristics encoded through Snorkel can provide a useful global training signal when gold labels are scarce.

---

### Chapter 6 — Stage 2: Few-Shot Adaptation and Meta-Learning

Purpose: Show how the Stage 1 representation is adapted to department-specific gold labels.

Include:
- Why departments are treated as tasks.
- Support/query episode construction.
- Logistics as target department.
- Source department validity.
- Zero-shot baseline.
- Fine-tuning baseline.
- ANIL.
- FOMAML.
- Full MAML.
- K-shot sweeps.
- Inner-step sweeps.
- Repeated-episode mean ± standard deviation.

Learning objectives covered:
- Train and evaluate meta-learning models.
- Use appropriate validation strategies.
- Critically assess assumptions and results.
- Align methodology with enterprise constraints.

Important caution:

> Full MAML requires multiple balanced source tasks to learn a generalizable meta-initialization. If the current gold-label inventory contains too few valid source departments, results should be interpreted as constrained adaptation benchmarking rather than a definitive claim about MAML superiority.

Suggested wording:

> Stage 2 evaluates whether department-specific adaptation improves over the weak-pretrained global model. The comparison against standard fine-tuning is essential because both methods receive the same few-shot support examples.

---

### Chapter 7 — Model Evaluation and Robustness

Purpose: Make the empirical evaluation credible.

Include:
- Metrics:
  - AUROC
  - Average Precision
  - log-loss
  - Brier score
  - ECE
  - Precision@K
  - NDCG@K
- Repeated support/query splits.
- K-shot curves.
- Inner-step curves.
- Probability distribution diagnostics.
- Calibration reliability plots.
- Sensitivity to class imbalance.
- Discussion of label acquisition needs.

Learning objectives covered:
- Apply appropriate ML methods and critically evaluate assumptions.
- Communicate analytical findings clearly.
- Evaluate decision-support relevance.

Important metric framing:
- AUROC: ranking ability across thresholds.
- Log-loss: probability quality and overconfidence.
- ECE: calibration and trustworthiness.
- NDCG@K: top-K prioritization quality.
- Precision@K: practical review efficiency.

Suggested wording:

> Since the business use case is prioritization rather than fully automated classification, ranking metrics such as Precision@K and NDCG@K are central. Calibration metrics are included because procurement users may interpret model scores as degrees of renegotiation pressure.

---

### Chapter 8 — Explainability and Stakeholder Interpretation

Purpose: Show whether model outputs are understandable and useful for decision-making.

Include:
- Global feature importance.
- SHAP summary plots.
- SHAP before vs after adaptation.
- View-level SHAP aggregation.
- Logistics-specific driver discussion.
- Embedding projection diagnostics, if useful:
  - department/task clustering
  - pre/post adaptation representation shift
  - gold-label separability

Learning objectives covered:
- Assess SHAP and feature importance for stakeholder insights.
- Communicate findings to non-technical stakeholders.
- Relate model outputs to procurement logic.

Important caution:

> SHAP values and embedding projections should be treated as interpretability diagnostics, not causal explanations.

Suggested wording:

> Explainability is evaluated by whether the highlighted drivers are plausible and actionable for procurement stakeholders, not merely by whether they improve model performance.

---

### Chapter 9 — Production Readiness and Replicability

Purpose: Show that the project is more than isolated notebooks.

Include:
- Pipeline structure:
  - data preparation
  - feature engineering
  - weak labeling
  - Stage 1 training
  - Stage 2 adaptation
  - evaluation
  - explainability
- Saved artifacts:
  - processed datasets
  - preprocessor
  - model weights
  - config files
  - metrics CSVs
  - diagnostic reports
  - pruning audit
- Reproducibility:
  - random seeds
  - run metadata
  - saved configurations
  - notebook order
  - modular `src/master_thesis` scripts
- Enterprise constraints:
  - confidential data
  - interpretability
  - maintainability
  - deployment limitations
  - label acquisition workflow

Learning objectives covered:
- Assess production readiness and replicability.
- Cover end-to-end modeling pipeline.
- Address enterprise deployment constraints.

Suggested wording:

> The pipeline is designed as a reproducible research artifact rather than a one-off experiment. Each stage saves intermediate outputs and validation checks so that downstream failures can be traced to specific data, labeling, or modeling assumptions.

---

### Chapter 10 — Discussion

Purpose: Critically interpret the results.

Include:
- What worked:
  - weak supervision as scalable signal generation
  - MLP as transferable representation
  - top-K prioritization
  - interpretability workflow
- What was limited:
  - few valid source departments
  - class imbalance in gold labels
  - Logistics label imbalance if unresolved
  - noisy weak labels
  - structured missingness
  - possible temporal limitations
- What the results mean if fine-tuning beats MAML.
- What the results mean if MAML improves after additional labels.
- Practical implications for procurement.

Learning objectives covered:
- Critical assessment.
- Communication of research-based knowledge.
- Applied business interpretation.

Important thesis-safe conclusion:

> The thesis does not assume that meta-learning is always superior. Instead, it evaluates under which data conditions meta-learning adds value beyond weak-pretrained fine-tuning.

---

### Chapter 11 — Conclusion and Future Work

Purpose: Summarize the framework and future research direction.

Include:
- Main findings.
- Contribution to decision-support analytics.
- Value of weak supervision.
- Conditions required for successful meta-learning.
- Future work:
  - more balanced gold labels
  - active learning
  - time-based validation
  - view-specific models
  - OOF stacking
  - calibration post-processing
  - deployment prototype

Learning objectives covered:
- Communicate findings clearly.
- Reflect on limitations and future development.

---

## 3. Objective-to-Evidence Alignment Table

Include a table like this in the thesis, either in the methodology chapter or appendix.

| Learning Objective | How the Thesis Addresses It | Evidence / Artifact |
|---|---|---|
| Identify and critically assess international research in ensemble learning, multi-view learning, data-driven decision support | Literature review covers weak supervision, multi-view learning, meta-learning, calibration, ranking, and explainability | Literature Review, Research Gap section |
| Analyze complex, heterogeneous data for procurement decision-making | Six-view dataset with contract, financial, ESG, news, market, and macro-logistics signals | Data diagnostics, missingness plots, feature view summary |
| Manage and prepare large-scale multi-source datasets | Feature engineering pipeline joins, cleans, validates, and saves processed datasets | Feature engineering notebook, processed feature table |
| Handle missing data, update frequencies, and time consistency | View-level missingness, missingness flags, `observation_year`, leakage screening, time-aware discussion | Missingness figures, leakage audit, pruning audit |
| Apply statistical and ML methods in Python | Snorkel, Elastic Net, XGBoost, MLP, fine-tuning, ANIL, FOMAML, MAML | Stage 1 and Stage 2 notebooks |
| Critically evaluate assumptions and results | Repeated episodes, k-shot sweeps, calibration, ranking, limitations discussion | Stage 2 benchmark, robustness checks |
| Communicate research-based findings clearly | Decision-support framing, visual roadmap, stakeholder interpretation of SHAP | Figures, roadmap, discussion |
| Develop modular multi-view / multi-source modeling framework | Six data views engineered into interpretable contract-level signals, optionally compared through view-specific models | Feature groups, view-level SHAP, view-specific baseline experiment |
| Assess leakage risk in multi-source time-dependent data | Leakage screening, protected columns, pruning audit, downstream validation | Feature pruning audit, leakage check |
| Train and evaluate stacked/meta-learning models | Weak-pretrained MLP, fine-tuning, ANIL, FOMAML, MAML, initialization ablations | Stage 2 experiments |
| Use appropriate validation strategies | Group-based splits, repeated support/query episodes, k-shot and inner-step sweeps, proposed time-based split | Evaluation chapter |
| Assess explainability methods | SHAP before/after adaptation, view-level importance, feature importance comparison | Explainability chapter |
| Assess production readiness and replicability | Modular scripts, configs, saved models, saved metrics, audit files, run metadata | Pipeline architecture and reproducibility section |

---

## 4. Recommended Additions to Strengthen Alignment

### 4.1 Add a View-Specific Baseline Experiment

To better satisfy the “specialized models per data view” objective, add a small experiment:

- contract-core-only model
- financial-only model
- ESG-only model
- news-only model
- market-only model
- macro-logistics-only model
- all-views model

Report:
- AUROC
- log-loss
- ECE
- NDCG@10

This creates direct evidence for multi-view signal fusion.

Suggested thesis wording:

> The view-specific baseline experiment evaluates whether individual data views carry independent predictive signal and whether the combined all-view representation improves decision-support performance.

---

### 4.2 Add a Time-Based Validation Discussion or Experiment

If possible, split by observation year:

- train on earlier years
- validate/test on later years

If the label count is too low, discuss it as a limitation and use time-consistency checks instead.

Suggested thesis wording:

> Due to limited gold-label availability, strict time-based validation is reported as a sensitivity analysis rather than the primary evaluation design. The primary evaluation uses group-based and episodic splits, while feature construction is constrained by observation-year availability to reduce leakage risk.

---

### 4.3 Clarify Stacking vs Signal Fusion

If you do not implement classical out-of-fold stacking, avoid overclaiming.

Use:

> signal fusion with meta-learning

instead of:

> full stacked generalization

Suggested wording:

> The framework is inspired by stacked and meta-learning principles but is implemented as weak-supervised signal fusion followed by neural few-shot adaptation. Classical out-of-fold stacking is treated as a related extension rather than the central experimental design.

---

### 4.4 Add Run Metadata Tracking

Every Stage 2 run should save:

- run ID
- timestamp
- random seed
- k-shot setup
- inner steps
- inner learning rate
- outer learning rate
- meta iterations
- target department
- source departments
- model initialization
- output path

This supports replicability.

---

## 5. How to Write Results Without Overclaiming

Use cautious, evidence-based language.

Avoid:

> MAML solves contract renegotiation prediction.

Use:

> MAML-style adaptation is evaluated as a candidate mechanism for department-specific specialization under label scarcity.

Avoid:

> Weak supervision creates true labels.

Use:

> Weak supervision creates probabilistic training targets derived from noisy domain heuristics.

Avoid:

> SHAP explains causality.

Use:

> SHAP provides feature-attribution evidence that supports qualitative interpretation of model behavior.

Avoid:

> The model predicts which contracts must be renegotiated.

Use:

> The model ranks contracts by estimated renegotiation pressure to support expert prioritization.

---

## 6. Minimum Evidence Needed for Each Learning Objective

Before final submission, make sure the thesis contains the following.

### Generic Objectives

- Literature review connecting methods to business problem.
- Data diagnostics for heterogeneous multi-source data.
- Feature engineering and missingness handling.
- Python-based modeling pipeline.
- Critical result interpretation.
- Clear figures and stakeholder-oriented explanations.

### Specific Objectives

- Multi-view feature framework and view-level analysis.
- Leakage and time-consistency subsection.
- Stage 1 and Stage 2 modeling experiments.
- Meta-learning/fine-tuning comparison.
- Validation strategy explanation.
- SHAP and feature-importance analysis.
- Production-readiness and reproducibility section.

---

## 7. Recommended Final Thesis Narrative

A strong final narrative is:

> This thesis develops and evaluates an end-to-end signal fusion framework for contract renegotiation prioritization under realistic enterprise constraints. Weak supervision is used to convert procurement heuristics into scalable probabilistic labels, enabling global representation learning across thousands of contract-year observations. Few-shot adaptation methods are then evaluated as mechanisms for department-specific specialization under scarce gold labels. The empirical results are interpreted through classification, ranking, calibration, and explainability metrics, while data diagnostics and leakage controls ensure that the framework remains suitable for decision-support use in a production-oriented procurement context.

---

## 8. Final Checklist

Before submitting, verify that the thesis includes:

- [ ] Data landscape overview.
- [ ] Gold-label scarcity and class-balance analysis.
- [ ] Multi-view missingness analysis.
- [ ] Feature engineering justification.
- [ ] Leakage screening and pruning audit.
- [ ] Snorkel LF diagnostics.
- [ ] Stage 1 baseline comparison.
- [ ] Stage 2 fine-tuning vs ANIL/FOMAML/MAML comparison.
- [ ] K-shot and inner-step sensitivity analysis.
- [ ] Calibration and ranking metrics.
- [ ] SHAP explainability.
- [ ] Production-readiness discussion.
- [ ] Objective-to-evidence alignment table.
- [ ] Careful limitations section.
