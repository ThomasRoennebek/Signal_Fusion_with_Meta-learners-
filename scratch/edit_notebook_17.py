import json
from pathlib import Path

notebook_path = Path("/Users/Thomas/Desktop/Master Thesis/notebooks/17_stage2_meta_learning.ipynb")

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# --- Update Diagnostic Cell (id: 675b6099) ---
for cell in nb["cells"]:
    if cell.get("id") == "675b6099":
        cell["source"] = [
            "# --- [STAGE 2] LINEAGE & FEATURE DIAGNOSTICS ---\n",
            "print(\"Stage 2 Execution Context:\")\n",
            "print(f\"  - Config:  {CONFIG_PATH.name}\")\n",
            "print(f\"  - Dataset: {config['data']['data_filename']}\")\n",
            "print(f\"  - Init Model: {config['stage1_init']['init_name']}\")\n",
            "\n",
            "try:\n",
            "    from master_thesis.data_utils import load_processed\n",
            "    df_diag = load_processed(config['data']['data_filename'])\n",
            "    \n",
            "    mandatory_features = [\n",
            "        \"financial_data_available\", \"esg_data_available\", \n",
            "        \"news_data_available\", \"market_data_available\",\n",
            "        \"has_environmental_appendix\"\n",
            "    ]\n",
            "    \n",
            "    print(\"\\nPillar Availability Flags (Presence in CSV):\")\n",
            "    for feat in mandatory_features:\n",
            "        exists = feat in df_diag.columns\n",
            "        status = \"✅ PASSED\" if exists else \"❌ MISSING (Check Notebook 14)\"\n",
            "        print(f\"  - {feat:28}: {status}\")\n",
            "    \n",
            "    print(\"\\nLabel Verification:\")\n",
            "    print(f\"  - gold_y present:           {'gold_y' in df_diag.columns}\")\n",
            "    print(f\"  - renegotiation_prob present: {'renegotiation_prob' in df_diag.columns}\")\n",
            "    \n",
            "    if \"label_source\" in df_diag.columns:\n",
            "        print(\"\\nGold Label Source Breakdown:\")\n",
            "        print(df_diag[\"label_source\"].value_counts())\n",
            "        \n",
            "except Exception as e:\n",
            "    print(f\"Diagnostic error: {e}\")\n"
        ]
        break

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Successfully updated Notebook 17.")
