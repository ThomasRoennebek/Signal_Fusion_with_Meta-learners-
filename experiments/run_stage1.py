import yaml
import argparse
from pathlib import Path

# from master_thesis.data_utils import load_raw, save_processed
# from master_thesis.snorkel_lfs import prepare_snorkel_dataframe
# from master_thesis.baselines import fit_xgboost_regressor

def main(config_path: str):
    """
    Orchestrates the Stage 1 pipeline:
    Loading data -> Applying Snorkel LFs -> Training Baselines -> Tracking metrics
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    print(f"Running Stage 1 Pipeline with config: {config_path}")
    
    # 1. Load Data
    # df = load_raw(...)
    
    # 2. Apply Snorkel Labeling Functions
    # df = prepare_snorkel_dataframe(df)
    # applier = PandasLFApplier(lfs=ALL_LFS)  # from snorkel_lfs
    # L_train = applier.apply(df=df)
    
    # 3. Fit Baseline Models
    # model, preds, results = fit_xgboost_regressor(...)
    # print(results)
    
    # 4. Save results/weak labels for Stage 2
    # save_processed(df_with_weak_labels, config['data']['processed_output_path'])
    
    print("Stage 1 complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Stage 1 Pipeline")
    parser.add_argument("--config", type=str, default="experiments/stage1_config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    main(args.config)
