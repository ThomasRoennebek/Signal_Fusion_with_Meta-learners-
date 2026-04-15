import json
from pathlib import Path

def patch_notebook():
    path = Path('/Users/Thomas/Desktop/Master Thesis/notebooks/13_prepare_gold_labels.ipynb')
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        # Ensure 'source' is a list of strings if that's how it's stored, or a single string
        source_content = "".join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        
        # Export Authoritative labels logic
        if 'df_gold_clean = (' in source_content and 'print("Gold labels after duplicate resolution:", len(df_gold_clean))' in source_content:
             new_source = source_content.replace(
                 'print("Gold labels after duplicate resolution:", len(df_gold_clean))',
                 '# [NEW] EXPORT AUTHORITATIVE GOLD LABELS\n# This file contains ONLY the human-reviewed / hardcoded labels.\ndf_gold_clean.to_csv(DATA_PROCESSED / "gold_labels_clean.csv", index=False)\n\nprint("Gold labels after duplicate resolution:", len(df_gold_clean))\nprint(f"✅ Exported authoritative labels to: {DATA_PROCESSED / \'gold_labels_clean.csv\'}")'
             )
             cell['source'] = new_source
        
        # Debug augmentation logic
        if '# --- ADDING DUMMY LABELS TO MASTER LIST ---' in source_content:
             cell['source'] = """# --- [DEBUG MODE] DETERMINISTIC FILLER AUGMENTATION ---
# This section ensures Stage 2 is runnable by padding Logistics to 10 pos / 10 neg.
# It uses deterministic selection (highest/lowest prob) if available, fallback to random.
import numpy as np
from datetime import date
np.random.seed(42)

TARGET_K = 10
DEBUG_DEPT = "Logistics"
dummy_rows = []

if DEBUG_DEPT in df_features['department'].values:
    dept_manual = df_gold_clean[df_gold_clean['department'] == DEBUG_DEPT]
    existing_ids = set(df_gold_clean['contract_id'])
    
    # Try to find probabilities for deterministic selection
    prob_col = 'renegotiation_prob'
    has_probs = prob_col in df_features.columns
    
    for label_val in [0, 1]:
        current_count = len(dept_manual[dept_manual['gold_y'] == label_val])
        if current_count < TARGET_K:
            needed = TARGET_K - current_count
            
            # Find candidate contracts in Logistics not already labeled
            candidates_df = df_features[
                (df_features['department'] == DEBUG_DEPT) & 
                (~df_features['contract_id'].astype(str).isin(existing_ids))
            ].copy()
            
            if not candidates_df.empty:
                if has_probs:
                    # Deterministic selection: highest prob for pos, lowest for neg
                    candidates_df = candidates_df.sort_values(prob_col, ascending=(label_val == 0))
                    sampled = candidates_df.head(needed)['contract_id'].unique()
                    selection_method = f"deterministic (via {prob_col})"
                else:
                    # Fallback: Random sampling
                    candidates = candidates_df['contract_id'].unique()
                    sampled = np.random.choice(candidates, size=min(needed, len(candidates)), replace=False)
                    selection_method = "random (renegotiation_prob missing)"
                
                for cid in sampled:
                    dummy_rows.append({
                        'contract_id': str(cid),
                        'department': DEBUG_DEPT,
                        'gold_y': label_val,
                        'label_source': 'DEBUG_FILLER',
                        'label_date': str(date.today()),
                        'notes': f'Augmented via {selection_method}'
                    })
                    existing_ids.add(str(cid))

# Add them to the augmented list
df_gold_augmented = pd.concat([df_gold_clean, pd.DataFrame(dummy_rows)], ignore_index=True)

# [NEW] EXPORT DEBUG AUGMENTED LABELS
df_gold_augmented.to_csv(DATA_PROCESSED / "gold_labels_debug_augmented.csv", index=False)

# [NEW] INTEGRITY CHECK: Clean file must have zero fillers
clean_check = pd.read_csv(DATA_PROCESSED / "gold_labels_clean.csv")
if 'label_source' in clean_check.columns:
    filler_count = (clean_check['label_source'] == 'DEBUG_FILLER').sum()
    if filler_count > 0:
        raise ValueError(f"CRITICAL ERROR: {filler_count} DEBUG_FILLER rows found in gold_labels_clean.csv!")

print(f"✅ Exported augmented labels to: {DATA_PROCESSED / 'gold_labels_debug_augmented.csv'}")
print(f"   (Added {len(dummy_rows)} fillers for {DEBUG_DEPT})")"""

    final_cell_content = """## 8. Export Joined Stage 2 Debug Dataset

This cell creates the final joined dataset required for Stage 2 development. It merges the features with the debug-augmented labels.

# Use the augmented set for the Stage 2 join
df_stage2_joined = df_features.merge(
    df_gold_augmented[['contract_id', 'gold_y', 'label_source']],
    on='contract_id',
    how='left'
)

# Fill missing source for weak-labeled rows
df_stage2_joined['label_source'] = df_stage2_joined['label_source'].fillna('none')

OUTPUT_JOINED_PATH = DATA_PROCESSED / "contract_with_features_gold_joined.csv"
df_stage2_joined.to_csv(OUTPUT_JOINED_PATH, index=False)

print(f"✅ Exported joined Stage 2 dataset to: {OUTPUT_JOINED_PATH}")
print(f"   Shape: {df_stage2_joined.shape}")
print(f"   Logistics gold count: {df_stage2_joined[df_stage2_joined['department']=='Logistics']['gold_y'].notna().sum()}")"""

    # Check if a cell with "Export Joined Stage 2 Debug Dataset" exists
    found_final = False
    for cell in nb['cells']:
        source_content = "".join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if "Export Joined Stage 2 Debug Dataset" in source_content:
            cell['source'] = final_cell_content
            found_final = True
            break
    
    if not found_final:
        nb['cells'].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 8. Export Joined Stage 2 Debug Dataset\n\nThis cell creates the final joined dataset required for Stage 2 development. It merges the features with the debug-augmented labels."]
        })
        nb['cells'].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Use the augmented set for the Stage 2 join\n",
                "df_stage2_joined = df_features.merge(\n",
                "    df_gold_augmented[['contract_id', 'gold_y', 'label_source']],\n",
                "    on='contract_id',\n",
                "    how='left'\n",
                ")\n",
                "\n",
                "# Fill missing source for weak-labeled rows\n",
                "df_stage2_joined['label_source'] = df_stage2_joined['label_source'].fillna('none')\n",
                "\n",
                "OUTPUT_JOINED_PATH = DATA_PROCESSED / \"contract_with_features_gold_joined.csv\"\n",
                "df_stage2_joined.to_csv(OUTPUT_JOINED_PATH, index=False)\n",
                "\n",
                "print(f\"✅ Exported joined Stage 2 dataset to: {OUTPUT_JOINED_PATH}\")\n",
                "print(f\"   Shape: {df_stage2_joined.shape}\")\n",
                "print(f\"   Logistics gold count: {df_stage2_joined[df_stage2_joined['department']=='Logistics']['gold_y'].notna().sum()}\")"
            ]
        })

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook patched successfully.")

if __name__ == "__main__":
    patch_notebook()
