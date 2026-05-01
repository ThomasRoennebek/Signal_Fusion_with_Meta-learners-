# Pipeline Stabilization & Alignment Changelog

## Root Cause
Stage 2 meta-learning was crashing with a `ValueError` because the Stage 1 preprocessor expected five mandatory feature columns (Availability Flags and Environmental Appendix) that were being generated in-memory during Stage 1 training but never persisted in the upstream dataset artifacts.

## Files Changed
- [14_feature_engineering.ipynb](file:///Users/Thomas/Desktop/Master%20Thesis/notebooks/14_feature_engineering.ipynb): Centralized flag generation and protected features.
- [16_stage1_baselines.ipynb](file:///Users/Thomas/Desktop/Master%20Thesis/notebooks/16_stage1_baselines.ipynb): Removed redundant in-memory flag recreation.
- [17_stage2_meta_learning.ipynb](file:///Users/Thomas/Desktop/Master%20Thesis/notebooks/17_stage2_meta_learning.ipynb): Added diagnostic cell to verify lineage.

## Exact Fix Applied
1. **Source Propagation**: Added `add_view_availability_flags()` to Notebook 14. This calculates `financial_data_available`, `esg_data_available`, `news_data_available`, and `market_data_available` as stable `Int64` columns in the base dataset.
2. **Feature Protection**: Added newly engineered flags to `THESIS_PROTECTED_FEATURES` so they are never pruned by correlation or missingness rules.
3. **Validation Refactoring**: Updated the final check in Notebook 14 to ignore \"future\" label columns (`renegotiation_prob`, `gold_y`) while strictly requiring the new engineer-level flags.
4. **Lineage Alignment**: Notebook 16 now consumes these pre-existing columns, ensuring the resulting preprocessor artifact is perfectly aligned with the shared file system.

## Required Rerun Order
To synchronize the repo, run the notebooks in this exact order:
1. `14_feature_engineering.ipynb`
2. `15_snorkel_labeling.ipynb`
3. `15.1_prepare_gold_labels.ipynb`
4. `16_stage1_baselines.ipynb` (Critical: generates the aligned preprocessor)
5. `17_stage2_meta_learning.ipynb`
