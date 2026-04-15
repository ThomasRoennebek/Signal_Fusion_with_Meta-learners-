import pandas as pd
import numpy as np
from pathlib import Path

DATA_PROCESSED = Path("/Users/Thomas/Desktop/Master Thesis/Data/processed")

def verify_pipeline():
    print("--- [VERIFICATION] PIPELINE ALIGNMENT ---")
    
    # Check if we can simulate the Notebook 13 join logic
    features_path = DATA_PROCESSED / "contracts_with_features.csv"
    if not features_path.exists():
        print(f"❌ Aborting verification: {features_path} missing.")
        return

    df_features = pd.read_csv(features_path, low_memory=False)
    print(f"✅ Loaded features: {len(df_features)} rows.")

    # Check for gold_labels_clean.csv (should be created by Notebook 13)
    # Since I haven't "run" the notebook, I'll check if my patch script logic works by running a snippet here.
    
    # (Simulating Notebook 13's df_gold_clean generation)
    gold_record_example = {'contract_id': '81', 'department': 'Legal', 'gold_y': 0} # Dummy for test
    df_gold_clean = pd.DataFrame([gold_record_example])
    
    # Simulating the Filler Logic
    TARGET_K = 2
    DEBUG_DEPT = "Logistics"
    dummy_rows = []
    
    if DEBUG_DEPT in df_features['department'].values:
        dept_rows = df_features[df_features['department'] == DEBUG_DEPT]
        print(f"Found {len(dept_rows)} rows in {DEBUG_DEPT}")
        
        for val in [0, 1]:
            sampled = dept_rows.head(TARGET_K)['contract_id'].unique()
            for cid in sampled:
                dummy_rows.append({
                    'contract_id': str(cid),
                    'department': DEBUG_DEPT,
                    'gold_y': val,
                    'label_source': 'DEBUG_FILLER'
                })
    
    df_gold_augmented = pd.concat([df_gold_clean, pd.DataFrame(dummy_rows)], ignore_index=True)
    print(f"✅ Created augmented gold set with {len(df_gold_augmented)} rows.")
    
    # Simulating the Join
    df_features['contract_id_str'] = df_features['contract_id'].astype(str)
    df_gold_augmented['contract_id_str'] = df_gold_augmented['contract_id'].astype(str)
    
    df_joined = df_features.merge(
        df_gold_augmented[['contract_id_str', 'gold_y', 'label_source']],
        on='contract_id_str',
        how='left'
    )
    
    print(f"✅ Joined dataset shape: {df_joined.shape}")
    if 'gold_y' in df_joined.columns and 'label_source' in df_joined.columns:
        print("✅ Columns 'gold_y' and 'label_source' are present.")
        label_counts = df_joined['label_source'].value_counts(dropna=False)
        print("Label counts:\n", label_counts)
    else:
        print("❌ Missing columns in join result.")

    # Check Config
    config_path = Path("/Users/Thomas/Desktop/Master Thesis/experiments/stage2_config.yaml")
    with open(config_path, 'r') as f:
        config_text = f.read()
    if 'data_filename: "contract_with_features_gold_joined.csv"' in config_text:
        print("✅ Config points to the correct joined file.")
    else:
        print("❌ Config filename mismatch.")

    print("\n--- VERIFICATION COMPLETE ---")

if __name__ == "__main__":
    verify_pipeline()
