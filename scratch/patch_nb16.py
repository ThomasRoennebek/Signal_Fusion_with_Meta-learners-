import json
from pathlib import Path

def patch_notebook():
    path = Path('/Users/Thomas/Desktop/Master Thesis/notebooks/16_stage1_baselines.ipynb')
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Find the data loading cell to insert diagnostic after it
    found_load_cell = -1
    for i, cell in enumerate(nb['cells']):
        source_content = "".join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if 'df_stage1 = load_processed(DATA_FILENAME' in source_content:
            found_load_cell = i
            break
    
    if found_load_cell != -1:
        diagnostic_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# --- [DEBUG MODE] STAGE 1 DATA DIAGNOSTICS ---\n",
                "print(\"Stage 1 Dataset Overview:\")\n",
                "print(f\"  - Total Rows: {len(df_stage1)}\")\n",
                "print(f\"  - Unique Contracts: {df_stage1[GROUP_COL].nunique()}\")\n",
                "\n",
                "if DEPARTMENT_COL in df_stage1.columns:\n",
                "    print(\"\\nDepartment Distribution (Top 10):\")\n",
                "    print(df_stage1[DEPARTMENT_COL].value_counts().head(10))\n",
                "    \n",
                "    if \"Logistics\" in df_stage1[DEPARTMENT_COL].values:\n",
                "        logistics_rows = len(df_stage1[df_stage1[DEPARTMENT_COL] == \"Logistics\"])\n",
                "        print(f\"\\n✅ Logistics department found with {logistics_rows} rows.\")\n",
                "    else:\n",
                "        print(\"\\n❌ WARNING: Logistics department NOT found in Stage 1 training data!\")\n",
                "\n",
                "print(f\"\\nWeak labels present ('{WEAK_TARGET_COL}'): {WEAK_TARGET_COL in df_stage1.columns}\")\n",
                "print(f\"Gold labels present ('{GOLD_COL}'): {GOLD_COL in df_stage1.columns}\")"
            ]
        }
        nb['cells'].insert(found_load_cell + 1, diagnostic_cell)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook 16 patched successfully.")

if __name__ == "__main__":
    patch_notebook()
