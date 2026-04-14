# Alignment Changelog

This package was aligned for the thesis pipeline:

1. Added `data_utils.py`, required by `16_stage1_baselines.ipynb`.
2. Updated `15_snorkel_labeling.ipynb` to save `contract_with_features_labeled.csv`, matching `stage1_config.yaml`.
3. Added downstream LF-compatible columns to processed CSVs where derivable:
   - `years_to_expiry_capped`
   - `days_until_expiry`
   - `lpi_below_supplier_median`
   - `is_old_and_near_expiry`
   - `esg_below_industry_min`
   - `supplier_is_publicly_listed`
   - `has_environmental_appendix`
4. Updated `17_stage2_meta_learning_updated.ipynb` to:
   - pass `meta_batch_size`
   - robustly extract metrics from live DataFrames, JSON records, or flat dictionaries
   - access MAML target episodes through `stage2_results["maml"]["prepared"]["target_episodes"]`
5. Updated `stage2.py` to save meta-training history for ANIL/FOMAML/MAML and serialize results more safely.
6. Set `stage2_config.yaml` to the safer 2-shot baseline:
   - `n_support_pos=2`
   - `n_support_neg=2`
   - `meta_batch_size=1`
   - `inner_lr=1e-4`
   - `outer_lr=5e-4`
   - `inner_steps=5`
   - `target_inner_steps=5`
7. Cleaned misleading inner-loop clipping comments in MAML/FOMAML/ANIL.
