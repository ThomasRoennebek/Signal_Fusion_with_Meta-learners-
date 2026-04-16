import json
from pathlib import Path

notebook_path = Path("/Users/Thomas/Desktop/Master Thesis/notebooks/16_stage1_baselines.ipynb")

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# The place to insert is after the markdown cell with id "abe71bc8" (## 2. Notebook Helpers)
idx_helpers_markdown = -1
for i, cell in enumerate(nb["cells"]):
    if cell.get("id") == "abe71bc8":
        idx_helpers_markdown = i
        break

if idx_helpers_markdown != -1:
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": "30285d07",
        "metadata": {},
        "outputs": [],
        "source": [
            "def get_stage1_feature_columns(\n",
            "    df_input: pd.DataFrame,\n",
            "    group_col: str,\n",
            "    weak_target_col: str,\n",
            "    gold_col: str,\n",
            ") -> list[str]:\n",
            "    leakage_cols = [\n",
            "        \"Unnamed: 0\",\n",
            "        group_col,\n",
            "        \"contract_number\",\n",
            "        \"contract_name\",\n",
            "        \"supplier_id\",\n",
            "        \"supplier_number\",\n",
            "        \"supplier_display_name\",\n",
            "        \"moodys_bvd_id\",\n",
            "        \"Company name Latin alphabet\",\n",
            "        \"company_name\",\n",
            "        \"gold_department\",\n",
            "        \"target_renegotiate\",\n",
            "        weak_target_col,\n",
            "        gold_col,\n",
            "        \"start_date\",\n",
            "        \"expiration_date\",\n",
            "        \"execution_at\",\n",
            "        \"published_at\",\n",
            "        \"contract_name_lower\",\n",
            "        \"lf_yes_votes\",\n",
            "        \"lf_no_votes\",\n",
            "        \"lf_abstain_votes\",\n",
            "        \"global_lifecycle_yes_votes\",\n",
            "        \"global_lifecycle_no_votes\",\n",
            "        \"global_financial_yes_votes\",\n",
            "        \"global_financial_no_votes\",\n",
            "        \"global_esg_yes_votes\",\n",
            "        \"global_esg_no_votes\",\n",
            "        \"global_news_yes_votes\",\n",
            "        \"global_news_no_votes\",\n",
            "        \"global_market_yes_votes\",\n",
            "        \"global_market_no_votes\",\n",
            "        \"global_supplier_macro_yes_votes\",\n",
            "        \"global_supplier_macro_no_votes\",\n",
            "        \"logistics_specific_yes_votes\",\n",
            "        \"logistics_specific_no_votes\",\n",
            "    ]\n",
            "    return [c for c in df_input.columns if c not in leakage_cols]\n",
            "\n",
            "\n",
            "def save_figure_versioned(fig, stem: str, output_dir: Path | None = None, dpi: int = 300) -> Path:\n",
            "    if output_dir is None:\n",
            "        output_dir = FIGURES\n",
            "\n",
            "    output_dir.mkdir(parents=True, exist_ok=True)\n",
            "    pattern = re.compile(rf\"^{re.escape(stem)}_v(\\d{{3}})\\.png$\")\n",
            "    versions = []\n",
            "\n",
            "    for path in output_dir.glob(f\"{stem}_v*.png\"):\n",
            "        match = pattern.match(path.name)\n",
            "        if match:\n",
            "            versions.append(int(match.group(1)))\n",
            "\n",
            "    next_version = max(versions, default=0) + 1\n",
            "    output_path = output_dir / f\"{stem}_v{next_version:03d}.png\"\n",
            "    fig.savefig(output_path, bbox_inches=\"tight\", dpi=dpi)\n",
            "    return output_path\n"
        ]
    }
    nb["cells"].insert(idx_helpers_markdown + 1, new_cell)

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Successfully restored missing functions to Notebook 16.")
