import json
from pathlib import Path

notebook_path = Path("/Users/Thomas/Desktop/Master Thesis/notebooks/14_feature_engineering.ipynb")

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# --- 1. Insert Availability Flags logic ---
# We'll insert it before the cell with id "559a1fe1" (Feature Pruning: Overview)
idx_pruning_overview = -1
for i, cell in enumerate(nb["cells"]):
    if cell.get("id") == "559a1fe1":
        idx_pruning_overview = i
        break

if idx_pruning_overview != -1:
    new_cells = [
        {
            "cell_type": "markdown",
            "id": "7829375e",
            "metadata": {},
            "source": [
                "## Data-View Availability Flags\n",
                "\n",
                "This section engineers four binary indicators that record whether a contract has available metadata for each major data view: Financial, ESG, News, and Market signals. These indicators are critical for the downstream Stage 1 and Stage 2 models, as they allow the neural network to learn whether a missing value represents an absence of signal or an absence of data coverage.\n",
                "\n",
                "Each flag is defined as a binary integer (`Int64`). A value of 1 indicates that at least one column from the respective view is populated for that contract, while 0 indicates a complete absence of data for that view."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "8293f75d",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Define and apply availability flags\n",
                "def add_view_availability_flags(df_input: pd.DataFrame) -> pd.DataFrame:\n",
                "    df_output = df_input.copy()\n",
                "\n",
                "    prefix_map = {\n",
                "        \"financial_data_available\": [\"fin_\"],\n",
                "        \"esg_data_available\": [\"esg_\"],\n",
                "        \"news_data_available\": [\"news_\"],\n",
                "    }\n",
                "\n",
                "    market_cols = [\n",
                "        \"avg_vol\", \"std_vol\", \"max_vol\", \"min_vol\", \"vol_stability_score\",\n",
                "        \"vol_shock_ratio\", \"vol_trend_slope\", \"avg_market_cap\",\n",
                "        \"market_cap_volatility\", \"market_beta_1y\", \"Earnings_per_share_DKK\",\n",
                "        \"Book_value_per_share_DKK\", \"avg_shares_outstanding\", \"avg_closing_price\",\n",
                "        \"price_volatility_score\", \"price_trend_slope\", \"Price_trends_52_weeks_pct\",\n",
                "        \"Price_trends_52 weeks_%\", \"moodys_risk_rating\",\n",
                "    ]\n",
                "\n",
                "    for new_col, prefixes in prefix_map.items():\n",
                "        cols = [c for c in df_output.columns if any(c.startswith(prefix) for prefix in prefixes)]\n",
                "        if cols: \n",
                "            df_output[new_col] = df_output[cols].notna().any(axis=1).astype(\"Int64\")\n",
                "        else:\n",
                "            df_output[new_col] = pd.Series([0]*len(df_output), dtype=\"Int64\")\n",
                "\n",
                "    market_existing = [c for c in market_cols if c in df_output.columns]\n",
                "    if market_existing:\n",
                "        df_output[\"market_data_available\"] = df_output[market_existing].notna().any(axis=1).astype(\"Int64\")\n",
                "    else:\n",
                "        df_output[\"market_data_available\"] = pd.Series([0]*len(df_output), dtype=\"Int64\")\n",
                "\n",
                "    # Cast environmental appendix to Int64 if present, filling nulls as 0\n",
                "    if \"has_environmental_appendix\" in df_output.columns:\n",
                "        df_output[\"has_environmental_appendix\"] = df_output[\"has_environmental_appendix\"].fillna(0).astype(\"Int64\")\n",
                "    else:\n",
                "        df_output[\"has_environmental_appendix\"] = pd.Series([0]*len(df_output), dtype=\"Int64\")\n",
                "\n",
                "    return df_output\n",
                "\n",
                "df_features = add_view_availability_flags(df_features)\n",
                "print(\"Engineered availability flags added to df_features.\")\n",
                "print(\"Columns:\", [c for c in df_features.columns if \"available\" in c or \"appendix\" in c])\n"
            ]
        }
    ]
    nb["cells"] = nb["cells"][:idx_pruning_overview] + new_cells + nb["cells"][idx_pruning_overview:]

# --- 2. Update THESIS_PROTECTED_FEATURES ---
for cell in nb["cells"]:
    if cell.get("id") == "ec292d4c":
        source = "".join(cell["source"])
        old_part = '    # Label columns'
        new_part = '    # Mandatory Stage 2 features\n    "financial_data_available", "esg_data_available", "news_data_available", "market_data_available",\n    "has_environmental_appendix",\n    # Label columns'
        if old_part in source:
            cell["source"] = source.replace(old_part, new_part).splitlines(keepends=True)
            # Ensure every line ends with \n except maybe the last one
            for j in range(len(cell["source"])-1):
                if not cell["source"][j].endswith("\n"): cell["source"][j] += "\n"
        break

# --- 3. Update Validation Cell ---
for cell in nb["cells"]:
    if cell.get("id") == "e55aa298":
        cell["source"] = [
            "saved_features = pd.read_csv(DATA_PROCESSED / \"contracts_with_features.csv\", nrows=5)\n",
            "required_downstream_columns = [\n",
            "    \"open_ended_contract\", \"days_until_expiry\", \"has_environmental_appendix\",\n",
            "    \"contracts_per_supplier\", \"years_to_expiry_capped\",\n",
            "    \"financial_data_available\", \"esg_data_available\", \"news_data_available\", \"market_data_available\",\n",
            "    \"esg_esg_industry_adjusted\", \"fin_flag_negative_ebit_margin\", \"fin_flag_multiple_financial_stress_signals\",\n",
            "    \"LPI_Score\", \"PPI_Value\", \"lpi_below_supplier_median\", \"is_old_and_near_expiry\",\n",
            "    \"esg_below_industry_min\", \"supplier_is_publicly_listed\", \"payment_terms\", \"incoterms\",\n",
            "    \"contract_age_years\", \"expiry_pressure_bucket\", \"contract_id\", \"supplier_id\",\n",
            "    \"department\", \"observation_year\"\n",
            "]\n",
            "missing_required = [c for c in required_downstream_columns if c not in saved_features.columns]\n",
            "if not missing_required:\n",
            "    print(\"All required downstream columns are present in contracts_with_features.csv.\")\n",
            "else:\n",
            "    # Separation of concerns: only error if features owned by this notebook are missing.\n",
            "    # renegotiation_prob and target_renegotiate are added later in Snorkel.\n",
            "    later_stage_cols = [\"target_renegotiate\", \"renegotiation_prob\", \"gold_y\"]\n",
            "    actual_missing = [c for c in missing_required if c not in later_stage_cols]\n",
            "    \n",
            "    if actual_missing:\n",
            "        print(\"ERROR: The saved feature matrix is missing downstream-required columns owned by this notebook:\")\n",
            "        for col in actual_missing:\n",
            "            print(f\" - {col}\")\n",
            "    else:\n",
            "        print(\"All engineer-level requirements are met.\")\n",
            "        print(\"Note: Label-level columns (renegotiation_prob, etc.) will be added in subsequent notebooks.\")\n"
        ]
        break

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Successfully updated Notebook 14.")
