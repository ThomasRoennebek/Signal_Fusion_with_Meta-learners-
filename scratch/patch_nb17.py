import json
from pathlib import Path

def patch_notebook():
    path = Path('/Users/Thomas/Desktop/Master Thesis/notebooks/17_stage2_meta_learning.ipynb')
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # 1. Insert verification/breakdown after config loading
    found_load_cell = -1
    for i, cell in enumerate(nb['cells']):
        source_content = "".join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if 'config = load_stage2_config(CONFIG_PATH)' in source_content:
            found_load_cell = i
            break
    
    if found_load_cell != -1:
        verification_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# --- [DEBUG MODE] CONFIG & DATA VERIFICATION ---\n",
                "print(\"Stage 2 Data Configuration:\")\n",
                "print(f\"  - Data Filename: {config['data']['data_filename']}\")\n",
                "print(f\"  - Target Department: {config['task_config']['target_department']}\")\n",
                "\n",
                "# Load internal dataset to check breakdown (simulating what run_stage2_pipeline does)\n",
                "try:\n",
                "    from master_thesis.data_utils import load_processed\n",
                "    df_debug = load_processed(config['data']['data_filename'])\n",
                "    print(\"\\nJoined Dataset Label Breakdown:\")\n",
                "    if 'label_source' in df_debug.columns:\n",
                "        print(df_debug['label_source'].value_counts())\n",
                "    else:\n",
                "        print(\"⚠️ 'label_source' column missing - check Notebook 13 output.\")\n",
                "    \n",
                "    logistics_mask = df_debug[config['data']['department_col']] == config['task_config']['target_department']\n",
                "    print(f\"\\nLogistics rows total: {logistics_mask.sum()}\")\n",
                "except Exception as e:\n",
                "    print(f\"Could not perform pre-run diagnostic: {e}\")"
            ]
        }
        nb['cells'].insert(found_load_cell + 1, verification_cell)

    # 2. Update metric extraction to be more robust
    for cell in nb['cells']:
        source_content = "".join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if 'def extract_stage2_metrics(stage2_results: dict)' in source_content:
            cell['source'] = [
                "def _coerce_metrics_to_frame(metrics, method: str) -> pd.DataFrame | None:\n",
                "    if metrics is None:\n",
                "        return None\n",
                "    if isinstance(metrics, pd.DataFrame):\n",
                "        if metrics.empty: return None\n",
                "        df = metrics.copy()\n",
                "    elif isinstance(metrics, list):\n",
                "        if len(metrics) == 0: return None\n",
                "        df = pd.DataFrame(metrics)\n",
                "    elif isinstance(metrics, dict):\n",
                "        flat = {k: v for k, v in metrics.items() if isinstance(v, (int, float, str, bool, np.number))}\n",
                "        if not flat: return None\n",
                "        df = pd.DataFrame([flat])\n",
                "    else:\n",
                "        return None\n",
                "    df[\"method\"] = method\n",
                "    return df\n",
                "\n",
                "def extract_stage2_metrics(stage2_results: dict) -> pd.DataFrame:\n",
                "    \"\"\"\n",
                "    Robust metric extraction helper [DEBUG MODE ALIGNED].\n",
                "    Handles cases where 'raw_metrics' might be missing if a method failed or was skipped.\n",
                "    \"\"\"\n",
                "    frames = []\n",
                "    for method, result in stage2_results.items():\n",
                "        # Handle both flat results and wrapped results\n",
                "        payload = result.get(\"result\", result) if isinstance(result, dict) else {}\n",
                "        target_eval = payload.get(\"target_eval_result\", {}) if isinstance(payload, dict) else {}\n",
                "        \n",
                "        # Ordered preference for data extraction\n",
                "        candidates = [\n",
                "            target_eval.get(\"raw_metrics\") if isinstance(target_eval, dict) else None,\n",
                "            target_eval.get(\"metrics\") if isinstance(target_eval, dict) else None,\n",
                "            payload.get(\"raw_metrics\") if isinstance(payload, dict) else None,\n",
                "            payload.get(\"metrics\") if isinstance(payload, dict) else None\n",
                "        ]\n",
                "        \n",
                "        df_method = None\n",
                "        for cand in candidates:\n",
                "            df_method = _coerce_metrics_to_frame(cand, method)\n",
                "            if df_method is not None:\n",
                "                break\n",
                "        \n",
                "        if df_method is not None:\n",
                "            frames.append(df_method)\n",
                "        else:\n",
                "            print(f\"❌ [ROBUST EXTRACTION] Missing metrics for {method}. Skipping.\")\n",
                "            \n",
                "    if not frames:\n",
                "        print(\"⚠️ No metrics found at all. Returning empty DataFrame to prevent crash.\")\n",
                "        return pd.DataFrame()\n",
                "        \n",
                "    return pd.concat(frames, ignore_index=True)\n",
                "\n",
                "df_stage2_metrics = extract_stage2_metrics(stage2_results)\n",
                "if not df_stage2_metrics.empty:\n",
                "    display(df_stage2_metrics.head())\n",
                "    print(\"Metric table shape:\", df_stage2_metrics.shape)\n",
                "else:\n",
                "    print(\"🛑 [CRITICAL] df_stage2_metrics is empty! Check Stage 2 run output above.\")\n",
                "\n",
                "extract_raw_metrics = extract_stage2_metrics # compatibility\n"
            ]

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook 17 patched successfully.")

if __name__ == "__main__":
    patch_notebook()
