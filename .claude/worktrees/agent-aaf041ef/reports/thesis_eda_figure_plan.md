# Thesis EDA and Figure Plan

This document defines the recommended figure set for the thesis project **Signal Fusion with Meta-Learners: An Explainable Framework for Data-Driven Decision Support**. The goal is not to include every exploratory plot, but to select figures that support the methodological argument: weak supervision is needed because gold labels are sparse, few-shot adaptation is difficult because labels are unevenly distributed across departments, and multi-view signal fusion is useful because contract renegotiation pressure is reflected across several partially observed data views.

## Figure selection principle

EDA figures should be included when they justify one of the following decisions:

1. Why weak supervision is needed.
2. Why multi-view signal fusion is useful.
3. Why missingness must be treated as structured information rather than random noise.
4. Why specific engineered features were created.
5. Why Stage 2 few-shot adaptation and meta-learning are difficult under the current gold-label distribution.

---

## A. Main thesis figures

### Figure 1 — Contract distribution by department

**Purpose:** Show the scale and imbalance of the full contract universe.

**Recommended plot:** Horizontal bar chart of unique contracts per department.

**Columns:** `department`, `contract_id`

**Thesis message:** The contract universe is unevenly distributed across procurement departments, which motivates department-aware analysis and evaluation.

**Suggested function:** `plot_department_contract_distribution()`

---

### Figure 2 — Gold label availability and class balance by department

**Purpose:** Show expert-label scarcity and imbalance.

**Recommended plot:** Stacked horizontal bar chart of `gold_y = 0` and `gold_y = 1` per department.

**Columns:** `department`, `gold_y`, `contract_id`

**Thesis message:** Gold labels are sparse and unevenly distributed. Some departments contain only one class, which limits standard supervised learning and full MAML meta-training.

**Suggested function:** `plot_gold_label_balance_by_department()`

---

### Figure 3 — Gold coverage rate by department

**Purpose:** Show the gap between available contracts and gold-labeled contracts.

**Recommended plot:** Horizontal bar chart of gold-label coverage percentage.

**Columns:** `department`, `contract_id`, `gold_y`

**Thesis message:** The number of unlabeled contracts is much larger than the number of expert-labeled contracts, directly motivating weak supervision.

**Suggested function:** `plot_gold_coverage_by_department()`

---

### Figure 4 — Multi-view missingness by data view

**Purpose:** Justify view-specific imputation and missingness-as-signal.

**Recommended plot:** Bar chart of average missingness by view.

**Views:** `contract_core`, `financial`, `esg`, `news`, `market`, `macro_logistics`, `labels`

**Thesis message:** Missingness is structured by data view, not random. For example, market data is often absent because most suppliers are private, while macro-logistics data is nearly complete.

**Suggested function:** `plot_view_missingness()` or `plot_view_missingness_summary()`

---

### Figure 5 — Snorkel renegotiation probability by department

**Purpose:** Show whether weak supervision produces useful variation.

**Recommended plot:** Boxplot or violin plot of `renegotiation_prob` by department.

**Columns:** `department`, `renegotiation_prob`

**Thesis message:** Snorkel produces non-constant renegotiation pressure scores with department-level variation, motivating department-specific adaptation.

**Suggested function:** `plot_snorkel_probability_distribution()`

---

### Figure 6 — Stage 2 task validity matrix

**Purpose:** Show which departments can support k-shot learning.

**Recommended plot:** Heatmap with departments as rows and k-shot settings as columns.

**Columns:** `department`, `valid_k2`, `valid_k5`, `valid_k10`

**Thesis message:** MAML is limited by department-level gold-label balance, not simply by the total number of labels.

**Suggested function:** `plot_stage2_task_validity_heatmap()`

---

## B. Feature-engineering justification figures

### Figure 7 — Financial stress feature correlation heatmap

**Source notebook:** `10_financials_eda.ipynb`

**Purpose:** Justify pruning raw financial values and keeping interpretable ratio/stress features.

**Recommended columns:**

- `fin_ebit_margin`
- `fin_current_ratio`
- `fin_gearing`
- `fin_interest_coverage_ratio`
- `fin_debt_asset_ratio`
- `fin_solvency_ratio_asset_based`
- `fin_financial_risk_score`
- `fin_total_stress_flags`
- `fin_flag_negative_ebit_margin`
- `fin_flag_multiple_financial_stress_signals`

**Thesis message:** Several financial indicators are highly correlated, so the final feature set prioritizes interpretable ratio-based and stress-flag features over redundant absolute balance-sheet values.

**Suggested function:** `plot_correlation_heatmap()`

---

### Figure 8 — Financial missingness distribution

**Source notebook:** `10_financials_eda.ipynb`

**Purpose:** Show why financial availability and missingness indicators matter.

**Recommended plot:** Histogram of financial row-level missingness or bar plot of financial feature missingness.

**Thesis message:** Financial data is informative but incomplete. Missingness may itself indicate supplier transparency or private-company data limitations.

**Suggested functions:** `plot_distribution()`, `plot_missingness_by_column()`

---

### Figure 9 — ESG score coverage and ESG missingness

**Source notebook:** `09_esg_eda.ipynb`

**Purpose:** Show that ESG is a sparse but specialist signal view.

**Recommended columns:**

- `esg_esg_industry_adjusted`
- `esg_env_score`
- `esg_social_score`
- `esg_gov_score`
- `esg_industry_min`
- `esg_industry_max`
- `esg_below_industry_min`

**Thesis message:** ESG coverage is limited, so ESG should be treated as a specialist signal rather than a universal driver.

**Suggested function:** `plot_missingness_by_column()`

---

### Figure 10 — ESG peer-relative threshold

**Source notebook:** `09_esg_eda.ipynb`

**Purpose:** Justify the feature `esg_below_industry_min`.

**Recommended plot:** Distribution of `esg_esg_industry_adjusted`, optionally faceted or annotated by peer-relative thresholds.

**Thesis message:** Peer-relative ESG signals are more interpretable than raw ESG scores because suppliers are compared against their own sector context.

**Suggested function:** `plot_distribution()`

---

### Figure 11 — LPI distribution and dataset-relative threshold

**Source notebook:** `11_indexes_eda.ipynb`

**Purpose:** Justify replacing a global LPI threshold with `lpi_below_supplier_median`.

**Recommended plot:** Histogram or density plot of `LPI_Score` with a vertical line at the supplier-country median.

**Columns:** `LPI_Score`, `lpi_below_supplier_median`, `company_country_clean`

**Thesis message:** A global academic LPI threshold is too low for the actual supplier population. A dataset-relative threshold better captures logistics risk within the supplier geography.

**Suggested function:** `plot_lpi_threshold_distribution()`

---

### Figure 12 — LPI sub-dimension correlation heatmap

**Source notebook:** `11_indexes_eda.ipynb`

**Purpose:** Show redundancy among logistics index dimensions.

**Recommended columns:**

- `LPI_Score`
- `Customs`
- `Infrastructure`
- `International_Shipments`
- `Logistics_Competence`
- `Tracking_Tracing`
- `Timeliness`

**Thesis message:** LPI sub-dimensions are strongly related, so not all dimensions contribute independent information. However, selected dimensions remain useful for interpretability.

**Suggested function:** `plot_correlation_heatmap()`

---

### Figure 13 — News coverage and sentiment distribution

**Source notebook:** `12_news_eda.ipynb`

**Purpose:** Justify news features as sparse event-driven signals.

**Recommended plots:**

1. Distribution of `news_article_count`
2. Distribution of `news_negative_ratio`
3. Distribution of `news_avg_sentiment_score`

**Thesis message:** News signals are sparse but useful for detecting event-driven supplier risk.

**Suggested function:** `plot_distribution()`

---

### Figure 14 — Publicly listed supplier coverage

**Source notebook:** `13_stocks_eda.ipynb`

**Purpose:** Justify `supplier_is_publicly_listed` and market-data missingness as signal.

**Recommended plot:** Bar chart of public vs private supplier records/contracts.

**Columns:** `supplier_is_publicly_listed`, `market_data_available`, `avg_closing_price`, `avg_vol`

**Thesis message:** Market data absence is not random; it indicates that most suppliers are privately held. Therefore, absence of stock data is itself informative.

**Suggested function:** `plot_public_supplier_coverage()`

---

## C. Stage 1 model diagnostic figures

### Figure 15 — Stage 1 gold-validity comparison

**Purpose:** Compare weak-only, gold-only, and hybrid training.

**Recommended plot:** Grouped bar chart.

**Metrics:** `gold_auroc`, `gold_logloss`, `gold_ece`, `ndcg_at_10`

**Thesis message:** Weak-supervised pretraining provides a scalable baseline, while gold-only training is unstable under label scarcity.

**Suggested function:** `plot_stage1_metric_comparison()`

---

### Figure 16 — Stage 1 prediction distribution

**Purpose:** Check whether models collapse to constant predictions.

**Recommended plot:** Density or histogram of model predictions by condition.

**Thesis message:** Prediction distributions help diagnose whether models produce meaningful risk variation or simply learn the class prior.

**Suggested function:** `plot_prediction_probability_distribution()`

---

### Figure 17 — Stage 1 calibration reliability diagram

**Purpose:** Show probability reliability.

**Recommended plot:** Reliability diagram.

**Thesis message:** Because the model supports decision prioritization, probability calibration matters in addition to ranking accuracy.

**Suggested function:** `plot_calibration_reliability()`

---

## D. Stage 2 meta-learning figures

### Figure 18 — Stage 2 method comparison

**Purpose:** Main Stage 2 result.

**Recommended plot:** Grouped bar chart with mean and standard deviation.

**Methods:** `zero-shot`, `fine-tune`, `ANIL`, `FOMAML`, `MAML`

**Metrics:** `AUROC`, `log-loss`, `ECE`, `NDCG@10`

**Thesis message:** The comparison evaluates whether meta-learning adds value beyond direct fine-tuning from the weak-pretrained MLP.

**Suggested function:** `plot_stage2_metric_comparison()`

---

### Figure 19 — Repeated episode distribution

**Purpose:** Show robustness and variance.

**Recommended plot:** Boxplot or violin plot.

**Metrics:** `AUROC`, `NDCG@10`, `log-loss`, `ECE`

**Thesis message:** Few-shot performance varies strongly depending on which contracts enter the support set, so repeated-episode distributions are more informative than single scores.

**Suggested function:** `plot_repeated_episode_boxplot()`

---

### Figure 20 — K-shot performance curve

**Purpose:** Core sample-efficiency figure for Stage 2.

**Recommended plot:** Line plot with k-shot size on the x-axis and performance on the y-axis.

**K values:** `2`, `5`, `10`

**Metrics:** `AUROC`, `NDCG@10`, `log-loss`

**Thesis message:** This shows how performance changes as more gold labels become available. If MAML improves only after k increases, that is a strong and honest result.

**Suggested function:** `plot_k_shot_curve()`

---

### Figure 21 — Inner-step adaptation curve

**Purpose:** Show whether adaptation helps or overfits.

**Recommended plot:** Line plot.

**X-axis:** `inner_steps = 1, 3, 5, 10`

**Y-axis:** `AUROC`, `query_loss`, or `NDCG@10`

**Thesis message:** Adaptation depth matters. Too few steps may under-adapt; too many steps may overfit the support set.

**Suggested function:** `plot_inner_step_curve()`

---

### Figure 22 — Prediction probability distribution after adaptation

**Purpose:** Diagnose overconfidence and calibration problems.

**Recommended plot:** Histogram or density plot of predicted probabilities by method.

**Thesis message:** High log-loss can occur when a model is overconfident even if ranking metrics are reasonable.

**Suggested function:** `plot_prediction_probability_distribution()`

---

## E. Explainability figures

### Figure 23 — SHAP before vs after adaptation

**Purpose:** Show how Logistics adaptation shifts model attention.

**Recommended plot:** Two SHAP summary plots or a delta-importance bar chart.

**Models:** Stage 1 weak-pretrained MLP vs post-adapted Logistics model.

**Thesis message:** Adaptation should change which features drive predictions, ideally toward Logistics-relevant contract and macro-logistics features.

**Suggested function:** `plot_shap_delta_importance()`

---

### Figure 24 — View-level SHAP importance

**Purpose:** Make SHAP easier to interpret at thesis level.

**Recommended plot:** Bar chart of feature importance aggregated by view.

**Views:** `contract_core`, `financial`, `esg`, `news`, `market`, `macro_logistics`

**Thesis message:** View-level importance shows which data sources contribute most to renegotiation prioritization.

**Suggested function:** `plot_view_level_shap_importance()`

---

## Main thesis vs appendix

### Main thesis figures

Use these in the main report:

1. Contract distribution by department.
2. Gold labels by department.
3. Gold coverage by department.
4. Multi-view missingness by view.
5. Snorkel probability distribution by department.
6. Stage 2 task validity matrix.
7. LPI dataset-relative threshold.
8. Stage 1 model comparison.
9. Stage 2 method comparison.
10. K-shot performance curve.
11. Repeated-episode boxplot.
12. SHAP before vs after adaptation.

### Appendix figures

Move these to appendix:

- Full financial correlation heatmap.
- ESG missingness details.
- News sentiment distributions.
- Stock market feature distributions.
- LPI sub-dimension correlation heatmap.
- All individual feature histograms.
- Full LF coverage table.
- Full pruning audit.

---

## Recommended figure numbering

1. Figure 1: Contract distribution across departments.
2. Figure 2: Gold-label availability and class balance by department.
3. Figure 3: Multi-view missingness profile.
4. Figure 4: Snorkel renegotiation probability by department.
5. Figure 5: Stage 2 task validity under different k-shot settings.
6. Figure 6: Dataset-relative LPI threshold construction.
7. Figure 7: Stage 1 baseline comparison.
8. Figure 8: Stage 2 adaptation benchmark.
9. Figure 9: K-shot sample-efficiency curve.
10. Figure 10: Repeated-episode robustness distribution.
11. Figure 11: SHAP feature attribution before and after Logistics adaptation.

This figure sequence gives a clear visual story: data reality → weak supervision → model comparison → adaptation → explainability.


---

## F. Representation and embedding diagnostics

These figures are optional but valuable because they connect the data landscape, weak supervision, and Stage 2 adaptation mechanism. They should be used as qualitative diagnostics, not as proof that a model performs well. The main evidence should still come from repeated-episode AUROC, log-loss, ECE, Precision@K, and NDCG@K.

### Figure 25 — Department/task structure in the learned representation

**Purpose:** Assess whether departments behave like distinct tasks in the feature space or in the Stage 1 MLP embedding space.

**Recommended plot:** PCA, t-SNE, or UMAP projection of model embeddings, colored by `department`.

**Data required:**

- encoded feature matrix or hidden-layer MLP embeddings
- `department`
- optionally `contract_id`

**Thesis message:** If departments form visible clusters, this supports the assumption that departments have different data distributions and may benefit from department-specific adaptation. If departments strongly overlap, it suggests that Stage 1 weak supervision may already capture a common contract representation and that fine-tuning may be sufficient.

**Suggested functions:**

- `compute_2d_projection()`
- `plot_department_embedding_projection()`

**Important interpretation rule:** This figure does not prove that MAML is necessary. It only diagnoses whether department/task heterogeneity is visible in the representation.

---

### Figure 26 — Gold-label separability before adaptation

**Purpose:** Check whether the Stage 1 weak-pretrained model already separates Logistics yes/no gold labels before any Stage 2 adaptation.

**Recommended plot:** PCA, t-SNE, or UMAP projection of Stage 1 embeddings for Logistics, colored by `gold_y`.

**Data required:**

- Stage 1 MLP embeddings for Logistics contracts
- `gold_y`
- `department`

**Thesis message:** If Logistics yes/no labels are already separated before adaptation, the weak-supervised Stage 1 model has learned a transferable representation and simple fine-tuning may be enough. If labels are mixed, Stage 2 adaptation has a clearer role.

**Suggested function:** `plot_gold_label_embedding_projection()`

**Important interpretation rule:** Separation in a projection is qualitative. It must be reported together with actual predictive metrics.

---

### Figure 27 — Representation shift before and after adaptation

**Purpose:** Show whether few-shot adaptation changes the model representation in a way that makes target-department labels more separable.

**Recommended plot:** Two-panel PCA, t-SNE, or UMAP projection fitted on the concatenated pre-adaptation and post-adaptation embeddings:

- Panel A: Stage 1 pre-adapted embeddings
- Panel B: post-adaptation embeddings
- Color: `gold_y`

**Data required:**

- pre-adaptation embeddings from the Stage 1 MLP
- post-adaptation embeddings from the adapted model
- `gold_y`

**Thesis message:** If pre-adaptation embeddings are mixed but post-adaptation embeddings separate target labels more clearly, this supports the claim that few-shot adaptation changes the internal representation toward department-specific renegotiation patterns.

**Suggested function:** `plot_pre_post_adaptation_projection()`

**Important interpretation rule:** This supports the adaptation story, but it does not by itself prove that MAML is better than fine-tuning. The comparison between adaptation methods must come from repeated metric distributions and k-shot curves.

---

### Figure 28 — Embedding shift magnitude

**Purpose:** Quantify how much each contract moves in representation space after adaptation.

**Recommended plot:** Histogram or boxplot of Euclidean shift magnitude between pre-adaptation and post-adaptation embeddings. Optionally split by `gold_y`.

**Data required:**

- pre-adaptation embeddings
- post-adaptation embeddings
- optionally `gold_y` or `department`

**Thesis message:** This figure helps diagnose whether adaptation changes the representation uniformly or whether it shifts one class/task subset more strongly than another.

**Suggested function:** `plot_embedding_shift_magnitude()`

---

## How representation plots answer the Stage 2 research question

These plots help answer the following diagnostic questions:

1. **Do departments form different task distributions?**  
   If yes, department-specific adaptation is conceptually justified.

2. **Does weak supervision already capture most department structure?**  
   If the Stage 1 embedding already separates departments or Logistics gold labels, then the weak-supervised pretraining step is doing most of the work.

3. **Does adaptation change the learned representation?**  
   If post-adaptation embeddings separate target labels better than pre-adaptation embeddings, adaptation is not merely changing the final probability threshold; it is changing the internal representation.

4. **Is meta-learning necessary, or is fine-tuning enough?**  
   Embedding plots can suggest why fine-tuning may perform well, but they cannot settle the question alone. The final answer must be based on repeated-episode metrics, k-shot curves, and calibration/ranking performance.

Recommended thesis wording:

> Embedding projections are used as qualitative diagnostics to assess department/task heterogeneity and representation shifts before and after adaptation. They are not treated as performance evidence, but as interpretability tools that help explain why weak-pretrained fine-tuning or meta-learning succeeds or fails under different label regimes.
