import yaml
import argparse
from pathlib import Path
import pandas as pd

# Now that all your heavy logic is cleanly inside src/master_thesis/stage1.py,
# you only need to import the high-level orchestration functions here!
from master_thesis.stage1 import (
    Stage1ModelConfig,
    fit_stage1_conditions,
    evaluate_stage1_conditions,
    save_stage1_condition_artifacts
)

def main(config_path: str):
    """
    Clean Execution Script for the Stage 1 pipeline.
    """
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
        
    print(f"🚀 Running Stage 1 Pipeline with config: {config_path}")
    
    # 1. Data Loading & Weak Labeling
    # (e.g. load_raw, prepare_snorkel_dataframe, apply LFs)
    # X_train, X_val, y_train, y_val = ...
    
    # 2. Prepare Config
    # config = Stage1ModelConfig(...)
    
    # 3. Define Conditions to test (e.g., 'A_weak_only', 'B_hybrid')
    # conditions = {
    #     "weak_only": {
    #         "X_train_df": X_train,
    #         "y_train": y_train,
    #         "sample_weight": None, 
    #     }
    # }
    
    # 4. Train Models
    # trained_bundles = fit_stage1_conditions(
    #     conditions=conditions,
    #     X_val=X_val,
    #     y_val=y_val,
    #     config=config,
    # )
    
    # 5. Evaluate and Save
    # For example, to save the 'weak_only' trial:
    # save_stage1_condition_artifacts(
    #     trained_bundle=trained_bundles["weak_only"], 
    #     output_dir=config_dict['data']['processed_output_path'],
    #     save_xgb_importance=True
    # )
    
    print("✅ Stage 1 complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Stage 1 Pipeline")
    parser.add_argument("--config", type=str, default="experiments/stage1_config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    main(args.config)