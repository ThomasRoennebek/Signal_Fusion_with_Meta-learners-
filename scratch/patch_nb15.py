import json
from pathlib import Path

def patch_notebook():
    path = Path('/Users/Thomas/Desktop/Master Thesis/notebooks/15_snorkel_labeling.ipynb')
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        source_content = "".join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        
        # Align Input Path in Cell 3
        if 'FEATURE_DATA_PATH =' in source_content and 'contracts_with_features.csv' in source_content:
             cell['source'] = [
                "INPUT_PATH = DATA_PROCESSED / \"contracts_with_features.csv\"\n",
                "\n",
                "df_features = pd.read_csv(INPUT_PATH, low_memory=False)\n",
                "\n",
                "print(\"Loaded:\", INPUT_PATH.name)\n",
                "print(\"Shape:\", df_features.shape)\n",
                "print(\"Unique contracts:\", df_features[\"contract_id\"].nunique())\n"
             ]

    # Add Validation Cell before "## Save outputs"
    found_save_header = -1
    for i, cell in enumerate(nb['cells']):
        source_content = "".join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if "## Save outputs" in source_content:
            found_save_header = i
            break
    
    if found_save_header != -1:
        validation_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# --- [DEBUG MODE] VALIDATION ---\n",
                "print(\"Labeled Dataset Statistics:\")\n",
                "print(f\"  - Total Rows: {len(df_snorkel)}\")\n",
                "print(f\"  - Unique Contracts: {df_snorkel['contract_id'].nunique()}\")\n",
                "\n",
                "if 'renegotiation_prob' in df_snorkel.columns:\n",
                "    print(\"✅ Found 'renegotiation_prob' column.\")\n",
                "else:\n",
                "    print(\"❌ WARNING: 'renegotiation_prob' missing!\")\n",
                "\n",
                "if 'target_renegotiate' in df_snorkel.columns:\n",
                "    print(\"✅ Found 'target_renegotiate' column.\")\n",
                "else:\n",
                "    print(\"❌ WARNING: 'target_renegotiate' missing!\")"
            ]
        }
        nb['cells'].insert(found_save_header, validation_cell)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook 15 patched successfully.")

if __name__ == "__main__":
    patch_notebook()
