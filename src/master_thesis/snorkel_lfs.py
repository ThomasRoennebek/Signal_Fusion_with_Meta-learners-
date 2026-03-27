"""
snorkel_lfs.py — Snorkel labeling functions for weak supervision.

Provides:
  - Label constants
  - Helper utilities safe_str, safe_lower, contains_any, is_logistics_department
  - prepare_snorkel_dataframe(df)
  - All labeling functions, grouped by signal category
  - LF_GROUPS
  - ALL_LFS
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from snorkel.labeling import labeling_function

# ── Label constants ────────────────────────────────────────────────────────
ABSTAIN = -1
NO = 0
YES = 1


# ── Helper utilities ───────────────────────────────────────────────────────

def safe_str(x) -> str:
    """Return x as a stripped string, or '' if NaN/None."""
    if pd.isna(x):
        return ""
    return str(x).strip()


def safe_lower(x) -> str:
    """Return lowercased safe_str(x)."""
    return safe_str(x).lower()


def contains_any(text, keywords) -> bool:
    """Return True if any keyword appears in the lowercased text."""
    text = safe_lower(text)
    return any(k.lower() in text for k in keywords)


def is_logistics_department(x) -> bool:
    """Return True if the department value refers to Logistics."""
    return "logistics" in safe_lower(x)


def safe_int_flag(value, default: int = 0) -> int:
    """
    Convert mixed flag values safely to 0/1.
    NaN or invalid values return default.
    """
    if pd.isna(value):
        return default

    if isinstance(value, str):
        value = value.strip().lower()
        mapping = {
            "true": 1,
            "false": 0,
            "yes": 1,
            "no": 0,
            "y": 1,
            "n": 0,
            "1": 1,
            "0": 0,
        }
        if value in mapping:
            return mapping[value]

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def prepare_snorkel_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a DataFrame for LF application.

    - adds lowercase helper columns
    - coerces boolean-like LF columns to 0/1
    - coerces numeric LF columns to numeric
    - creates contracts_per_supplier if missing
    - resets index for Snorkel row application
    """
    df = df.copy()

    if "contract_name" in df.columns:
        df["contract_name_lower"] = df["contract_name"].apply(safe_lower)

    bool_like_cols = [
        "open_ended_contract",
        "terminated",
        "news_has_high_relevance_negative_news",
        "fin_flag_weak_implied_rating",
        "fin_flag_moderate_or_worse_rating",
        "fin_flag_risk_take_caution_or_worse",
        "fin_flag_risk_do_not_source",
        "fin_flag_high_financial_risk_level",
        "fin_flag_financial_risk_score_high",
        "fin_flag_financial_risk_score_very_high",
        "fin_flag_liquidity_stress",
        "fin_flag_severe_liquidity_stress",
        "fin_flag_strong_liquidity",
        "fin_flag_gearing_high",
        "fin_flag_long_term_gearing_high",
        "fin_flag_short_term_gearing_high",
        "fin_flag_debt_asset_high",
        "fin_flag_debt_asset_very_high",
        "fin_flag_long_term_liab_equity_high",
        "fin_flag_short_term_liab_equity_high",
        "fin_flag_interest_coverage_stress",
        "fin_flag_interest_coverage_weak",
        "fin_flag_low_solvency",
        "fin_flag_very_low_solvency",
        "fin_flag_negative_profit_margin",
        "fin_flag_negative_ebit_margin",
        "fin_flag_profitability_stress",
        "fin_flag_negative_roe",
        "fin_flag_negative_roa",
        "fin_flag_multiple_financial_stress_signals",
        "fin_flag_severe_financial_stress",
        "market_flag_high_volume_shock",
        "market_flag_high_market_cap_volatility",
        "market_flag_negative_volume_trend",
        "market_flag_negative_price_trend",
        "market_flag_negative_52w_price_trend",
        "market_flag_high_beta",
        "market_flag_negative_eps",
        "market_flag_stock_price_take_caution_or_worse",
    ]

    for col in bool_like_cols:
        if col in df.columns:
            df[col] = df[col].apply(safe_int_flag)

    numeric_cols = [
        "days_until_expiry",
        "contract_age_years",
        "years_to_expiry",
        "news_article_count",
        "news_negative_count",
        "news_negative_ratio",
        "esg_esg_overall",
        "esg_esg_industry_adjusted",
        "esg_env_score",
        "esg_social_score",
        "esg_gov_score",
        "fin_total_stress_flags",
        "LPI_Score",
        "PPI_Value",
        "market_log_vol_shock_ratio",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "contracts_per_supplier" not in df.columns and "supplier_id" in df.columns:
        contracts_per_supplier = (
            df.groupby("supplier_id")["contract_id"]
            .nunique()
            .rename("contracts_per_supplier")
        )
        df = df.merge(
            contracts_per_supplier,
            left_on="supplier_id",
            right_index=True,
            how="left"
        )

    if "has_environmental_appendix" not in df.columns:
        df["has_environmental_appendix"] = np.nan

    df = df.reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════════
# Global lifecycle / expiry LFs
# ══════════════════════════════════════════════════════════════════════════

@labeling_function()
def lf_global_expired_contract(x):
    if pd.notna(x.days_until_expiry) and x.days_until_expiry < 0:
        return YES
    return ABSTAIN


@labeling_function()
def lf_global_near_expiry_contract(x):
    if pd.notna(x.days_until_expiry) and 0 <= x.days_until_expiry <= 180:
        return YES
    return ABSTAIN


@labeling_function()
def lf_global_old_perpetual_contract(x):
    if (
        safe_int_flag(getattr(x, "open_ended_contract", 0)) == 1
        and pd.notna(x.contract_age_years)
        and x.contract_age_years >= 5
    ):
        return YES
    return ABSTAIN


@labeling_function()
def lf_global_old_contract(x):
    if pd.notna(x.contract_age_years) and x.contract_age_years >= 7:
        return YES
    return ABSTAIN


@labeling_function()
def lf_global_expiry_pressure_bucket(x):
    bucket = safe_lower(getattr(x, "expiry_pressure_bucket", ""))
    if bucket in {"high", "critical", "urgent"}:
        return YES
    return ABSTAIN


# ══════════════════════════════════════════════════════════════════════════
# Global financial LFs
# ══════════════════════════════════════════════════════════════════════════

@labeling_function()
def lf_global_financial_severe_stress(x):
    if safe_int_flag(getattr(x, "fin_flag_severe_financial_stress", 0)) == 1:
        return YES
    return ABSTAIN


@labeling_function()
def lf_global_financial_multiple_stress(x):
    if safe_int_flag(getattr(x, "fin_flag_multiple_financial_stress_signals", 0)) == 1:
        return YES
    return ABSTAIN


@labeling_function()
def lf_global_financial_rating_weak(x):
    if safe_int_flag(getattr(x, "fin_flag_moderate_or_worse_rating", 0)) == 1:
        return YES
    return ABSTAIN


@labeling_function()
def lf_global_financial_do_not_source(x):
    if safe_int_flag(getattr(x, "fin_flag_risk_do_not_source", 0)) == 1:
        return YES
    return ABSTAIN


@labeling_function()
def lf_global_financial_liquidity_stress(x):
    if (
        safe_int_flag(getattr(x, "fin_flag_liquidity_stress", 0)) == 1
        or safe_int_flag(getattr(x, "fin_flag_severe_liquidity_stress", 0)) == 1
    ):
        return YES
    return ABSTAIN


@labeling_function()
def lf_global_financial_profitability_stress(x):
    if (
        safe_int_flag(getattr(x, "fin_flag_profitability_stress", 0)) == 1
        or safe_int_flag(getattr(x, "fin_flag_negative_profit_margin", 0)) == 1
        or safe_int_flag(getattr(x, "fin_flag_negative_ebit_margin", 0)) == 1
    ):
        return YES
    return ABSTAIN


@labeling_function()
def lf_global_financial_low_solvency(x):
    if (
        safe_int_flag(getattr(x, "fin_flag_low_solvency", 0)) == 1
        or safe_int_flag(getattr(x, "fin_flag_very_low_solvency", 0)) == 1
    ):
        return YES
    return ABSTAIN


@labeling_function()
def lf_global_financial_no_signal_strong_liquidity(x):
    if (
        safe_int_flag(getattr(x, "fin_flag_strong_liquidity", 0)) == 1
        and pd.notna(getattr(x, "fin_total_stress_flags", np.nan))
        and getattr(x, "fin_total_stress_flags") == 0
    ):
        return NO
    return ABSTAIN


# ══════════════════════════════════════════════════════════════════════════
# Global ESG LFs
# ══════════════════════════════════════════════════════════════════════════

@labeling_function()
def lf_global_esg_very_low_overall(x):
    if pd.notna(x.esg_esg_overall) and x.esg_esg_overall <= 2:
        return YES
    return ABSTAIN


@labeling_function()
def lf_global_esg_low_industry_adjusted(x):
    if pd.notna(x.esg_esg_industry_adjusted) and x.esg_esg_industry_adjusted <= 2:
        return YES
    return ABSTAIN


@labeling_function()
def lf_global_esg_low_env_or_social(x):
    env_bad = pd.notna(x.esg_env_score) and x.esg_env_score <= 2
    soc_bad = pd.notna(x.esg_social_score) and x.esg_social_score <= 2
    if env_bad or soc_bad:
        return YES
    return ABSTAIN


@labeling_function()
def lf_global_esg_low_governance(x):
    if pd.notna(x.esg_gov_score) and x.esg_gov_score <= 4:
        return YES
    return ABSTAIN


# ══════════════════════════════════════════════════════════════════════════
# Global news LFs
# ══════════════════════════════════════════════════════════════════════════

@labeling_function()
def lf_global_news_high_relevance_negative(x):
    if safe_int_flag(getattr(x, "news_has_high_relevance_negative_news", 0)) == 1:
        return YES
    return ABSTAIN


@labeling_function()
def lf_global_news_negative_ratio_high(x):
    if (
        pd.notna(getattr(x, "news_article_count", np.nan))
        and pd.notna(getattr(x, "news_negative_ratio", np.nan))
        and x.news_article_count >= 5
        and x.news_negative_ratio >= 0.20
    ):
        return YES
    return ABSTAIN


@labeling_function()
def lf_global_news_many_articles_and_negative(x):
    if (
        pd.notna(getattr(x, "news_article_count", np.nan))
        and pd.notna(getattr(x, "news_negative_count", np.nan))
        and x.news_article_count >= 10
        and x.news_negative_count >= 2
    ):
        return YES
    return ABSTAIN


@labeling_function()
def lf_global_news_positive_low_risk(x):
    if (
        pd.notna(getattr(x, "news_article_count", np.nan))
        and pd.notna(getattr(x, "news_negative_ratio", np.nan))
        and x.news_article_count >= 5
        and x.news_negative_ratio == 0
    ):
        return NO
    return ABSTAIN


# ══════════════════════════════════════════════════════════════════════════
# Global market LFs
# ══════════════════════════════════════════════════════════════════════════

@labeling_function()
def lf_global_market_high_stress(x):
    votes = [
        safe_int_flag(getattr(x, "market_flag_high_volume_shock", 0)),
        safe_int_flag(getattr(x, "market_flag_high_market_cap_volatility", 0)),
        safe_int_flag(getattr(x, "market_flag_negative_volume_trend", 0)),
        safe_int_flag(getattr(x, "market_flag_negative_price_trend", 0)),
        safe_int_flag(getattr(x, "market_flag_negative_52w_price_trend", 0)),
        safe_int_flag(getattr(x, "market_flag_high_beta", 0)),
        safe_int_flag(getattr(x, "market_flag_negative_eps", 0)),
        safe_int_flag(getattr(x, "market_flag_stock_price_take_caution_or_worse", 0)),
    ]
    if sum(votes) >= 2:
        return YES
    return ABSTAIN


@labeling_function()
def lf_global_market_negative_eps(x):
    if safe_int_flag(getattr(x, "market_flag_negative_eps", 0)) == 1:
        return YES
    return ABSTAIN


@labeling_function()
def lf_global_market_negative_price_trend(x):
    if safe_int_flag(getattr(x, "market_flag_negative_52w_price_trend", 0)) == 1:
        return YES
    return ABSTAIN


@labeling_function()
def lf_global_market_public_stress(x):
    if (
        safe_int_flag(getattr(x, "market_flag_stock_price_take_caution_or_worse", 0)) == 1
        and pd.notna(getattr(x, "market_log_vol_shock_ratio", np.nan))
        and x.market_log_vol_shock_ratio > np.log1p(10)
    ):
        return YES
    return ABSTAIN


# ══════════════════════════════════════════════════════════════════════════
# Global supplier / portfolio / macro LFs
# ══════════════════════════════════════════════════════════════════════════

@labeling_function()
def lf_global_many_contracts_same_supplier(x):
    if pd.notna(getattr(x, "contracts_per_supplier", np.nan)) and x.contracts_per_supplier >= 3:
        return YES
    return ABSTAIN


@labeling_function()
def lf_global_missing_environmental_appendix(x):
    val = getattr(x, "has_environmental_appendix", np.nan)
    if pd.notna(val) and safe_int_flag(val) == 0:
        return YES
    return ABSTAIN


@labeling_function()
def lf_global_open_ended_and_old_and_many_contracts(x):
    if (
        safe_int_flag(getattr(x, "open_ended_contract", 0)) == 1
        and pd.notna(getattr(x, "contract_age_years", np.nan))
        and x.contract_age_years >= 5
        and pd.notna(getattr(x, "contracts_per_supplier", np.nan))
        and x.contracts_per_supplier >= 2
    ):
        return YES
    return ABSTAIN


@labeling_function()
def lf_global_low_lpi(x):
    if pd.notna(getattr(x, "LPI_Score", np.nan)) and x.LPI_Score < 3.0:
        return YES
    return ABSTAIN


@labeling_function()
def lf_global_high_ppi_pressure(x):
    if pd.notna(getattr(x, "PPI_Value", np.nan)) and x.PPI_Value >= 120:
        return YES
    return ABSTAIN


# ══════════════════════════════════════════════════════════════════════════
# Logistics-specific LFs
# ══════════════════════════════════════════════════════════════════════════

@labeling_function()
def lf_logistics_expired_or_near_expiry(x):
    if not is_logistics_department(getattr(x, "department", "")):
        return ABSTAIN
    if pd.notna(x.days_until_expiry) and x.days_until_expiry <= 180:
        return YES
    return ABSTAIN


@labeling_function()
def lf_logistics_old_perpetual_contract(x):
    if not is_logistics_department(getattr(x, "department", "")):
        return ABSTAIN
    if (
        safe_int_flag(getattr(x, "open_ended_contract", 0)) == 1
        and pd.notna(x.contract_age_years)
        and x.contract_age_years >= 5
    ):
        return YES
    return ABSTAIN


@labeling_function()
def lf_logistics_low_lpi(x):
    if not is_logistics_department(getattr(x, "department", "")):
        return ABSTAIN
    if pd.notna(getattr(x, "LPI_Score", np.nan)) and x.LPI_Score < 3.2:
        return YES
    return ABSTAIN


@labeling_function()
def lf_logistics_term_misalignment_payment_terms(x):
    if not is_logistics_department(getattr(x, "department", "")):
        return ABSTAIN
    pt = safe_lower(getattr(x, "payment_terms", ""))
    if any(k in pt for k in ["f015", "f030", "15 days", "30 days"]):
        return YES
    return ABSTAIN


@labeling_function()
def lf_logistics_term_misalignment_incoterms(x):
    if not is_logistics_department(getattr(x, "department", "")):
        return ABSTAIN
    inc = safe_lower(getattr(x, "incoterms", ""))
    if inc in {"", "unknown"}:
        return ABSTAIN
    if any(k in inc for k in ["exw", "dap", "ddp", "fca", "cpt"]):
        return YES
    return ABSTAIN


@labeling_function()
def lf_logistics_supplier_fragmentation(x):
    if not is_logistics_department(getattr(x, "department", "")):
        return ABSTAIN
    if pd.notna(getattr(x, "contracts_per_supplier", np.nan)) and x.contracts_per_supplier >= 2:
        return YES
    return ABSTAIN


@labeling_function()
def lf_logistics_cost_driver_shift(x):
    if not is_logistics_department(getattr(x, "department", "")):
        return ABSTAIN
    market_stress_votes = [
        safe_int_flag(getattr(x, "market_flag_high_volume_shock", 0)),
        safe_int_flag(getattr(x, "market_flag_high_market_cap_volatility", 0)),
        safe_int_flag(getattr(x, "market_flag_negative_price_trend", 0)),
        safe_int_flag(getattr(x, "market_flag_negative_52w_price_trend", 0)),
    ]
    ppi_high = pd.notna(getattr(x, "PPI_Value", np.nan)) and x.PPI_Value >= 120
    if ppi_high or sum(market_stress_votes) >= 2:
        return YES
    return ABSTAIN


@labeling_function()
def lf_logistics_capacity_or_scope_proxy(x):
    if not is_logistics_department(getattr(x, "department", "")):
        return ABSTAIN
    text_fields = " ".join([
        safe_str(getattr(x, "contract_name", "")),
        safe_str(getattr(x, "contract_type", "")),
        safe_str(getattr(x, "nn_contract_type", "")),
        safe_str(getattr(x, "contract_commodity", "")),
    ]).lower()
    keywords = [
        "warehouse", "warehousing", "capacity", "forwarding", "distribution",
        "freight", "transport", "storage", "bonded", "scope", "logistics",
    ]
    if any(k in text_fields for k in keywords):
        return YES
    return ABSTAIN


@labeling_function()
def lf_logistics_outdated_contract_proxy(x):
    if not is_logistics_department(getattr(x, "department", "")):
        return ABSTAIN
    old_contract = pd.notna(x.contract_age_years) and x.contract_age_years >= 6
    perpetual = safe_int_flag(getattr(x, "open_ended_contract", 0)) == 1
    if old_contract or perpetual:
        return YES
    return ABSTAIN


# ══════════════════════════════════════════════════════════════════════════
# LF registry
# ══════════════════════════════════════════════════════════════════════════

LF_GROUPS: dict[str, list] = {
    "global_lifecycle": [
        lf_global_expired_contract,
        lf_global_near_expiry_contract,
        lf_global_old_perpetual_contract,
        lf_global_old_contract,
        lf_global_expiry_pressure_bucket,
    ],
    "global_financial": [
        lf_global_financial_severe_stress,
        lf_global_financial_multiple_stress,
        lf_global_financial_rating_weak,
        lf_global_financial_do_not_source,
        lf_global_financial_liquidity_stress,
        lf_global_financial_profitability_stress,
        lf_global_financial_low_solvency,
        lf_global_financial_no_signal_strong_liquidity,
    ],
    "global_esg": [
        lf_global_esg_very_low_overall,
        lf_global_esg_low_industry_adjusted,
        lf_global_esg_low_env_or_social,
        lf_global_esg_low_governance,
    ],
    "global_news": [
        lf_global_news_high_relevance_negative,
        lf_global_news_negative_ratio_high,
        lf_global_news_many_articles_and_negative,
        lf_global_news_positive_low_risk,
    ],
    "global_market": [
        lf_global_market_high_stress,
        lf_global_market_negative_eps,
        lf_global_market_negative_price_trend,
        lf_global_market_public_stress,
    ],
    "global_supplier_macro": [
        lf_global_many_contracts_same_supplier,
        lf_global_missing_environmental_appendix,
        lf_global_open_ended_and_old_and_many_contracts,
        lf_global_low_lpi,
        lf_global_high_ppi_pressure,
    ],
    "logistics_specific": [
        lf_logistics_expired_or_near_expiry,
        lf_logistics_old_perpetual_contract,
        lf_logistics_low_lpi,
        lf_logistics_term_misalignment_payment_terms,
        lf_logistics_term_misalignment_incoterms,
        lf_logistics_supplier_fragmentation,
        lf_logistics_cost_driver_shift,
        lf_logistics_capacity_or_scope_proxy,
        lf_logistics_outdated_contract_proxy,
    ],
}

ALL_LFS: list = [lf for group in LF_GROUPS.values() for lf in group]