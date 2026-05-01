"""
tests/test_synthetic_plots.py — Unit tests for synthetic_plots.py.

Tests cover:
- Plotting functions run on a tiny fake result DataFrame without errors.
- Summary table contains expected grouping columns.
- DTU palette constants are used (not random / untracked colours).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for CI

from master_thesis.synthetic_plots import (
    # DTU palette constants
    DTU_RED, DTU_BLUE, DTU_NAVY, DTU_BRIGHT_GREEN, DTU_ORANGE,
    DTU_YELLOW, DTU_GREY, DTU_DARK_GREY, DTU_PURPLE,
    # Internal mappings
    _METHOD_COLORS, _AUG_COLORS, _DTU_CYCLE,
    # Functions under test
    apply_dtu_style,
    get_dtu_palette,
    load_results,
    aggregate_by,
    build_summary_table,
    summarize_stage2b_results,
    plot_auroc_vs_real_k,
    plot_auroc_vs_syn_prop,
    plot_realism_vs_syn_prop,
    plot_stage2b_adaptation_curve,
    plot_stage2b_shot_matrix,
    REQUIRED_COLS,
    GROUP_COLS_BASE,
)


# ---------------------------------------------------------------------------
# Fixtures — tiny fake results DataFrame
# ---------------------------------------------------------------------------

def _make_fake_results(n_rows: int = 60) -> pd.DataFrame:
    """Build a tiny fake long-form results DataFrame that satisfies REQUIRED_COLS."""
    rng = np.random.RandomState(42)
    departments = ["dept_A", "dept_B"]
    methods = ["finetune", "anil", "fomaml"]
    gen_methods = ["none", "smote_nc"]
    init_names = ["A_weak_only", "C_hybrid"]
    real_ks = [1, 2, 3]

    rows = []
    for i in range(n_rows):
        rows.append({
            "department": departments[i % len(departments)],
            "method": methods[i % len(methods)],
            "init_name": init_names[i % len(init_names)],
            "real_k": real_ks[i % len(real_ks)],
            "gen_method": gen_methods[i % len(gen_methods)],
            "syn_prop": 0.0 if i % 2 == 0 else 0.5,
            "augmentation_method_actual": gen_methods[i % len(gen_methods)],
            "augmentation_method_requested": gen_methods[i % len(gen_methods)],
            "target_effective_k": None if i % 2 == 0 else 5,
            "auroc": rng.uniform(0.5, 0.9),
            "ap": rng.uniform(0.3, 0.8),
            "average_precision": rng.uniform(0.3, 0.8),
            "ndcg_at_10": rng.uniform(0.4, 0.9),
            "ece": rng.uniform(0.01, 0.15),
            "log_loss": rng.uniform(0.3, 0.7),
            "brier": rng.uniform(0.1, 0.4),
            "precision_at_10": rng.uniform(0.2, 0.7),
            "pred_mean": rng.uniform(0.3, 0.7),
            "pred_std": rng.uniform(0.05, 0.2),
            "collapsed": rng.choice([True, False], p=[0.05, 0.95]),
            "dist_to_nearest_real_mean": rng.uniform(0.1, 0.5),
            "dist_to_nearest_real_p95": rng.uniform(0.3, 0.8),
            "ks_max": rng.uniform(0.05, 0.4),
            "ks_median": rng.uniform(0.02, 0.2),
            "cat_mode_preservation_mean": rng.uniform(0.7, 1.0),
            "cat_mode_preservation_min": rng.uniform(0.5, 1.0),
            "n_real_support_rows": rng.randint(2, 10),
            "n_synthetic_support_rows": 0 if i % 2 == 0 else rng.randint(1, 10),
            "episode_idx": i,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def fake_results():
    return _make_fake_results()


# ---------------------------------------------------------------------------
# DTU palette constants
# ---------------------------------------------------------------------------

class TestDTUPaletteConstants:
    """DTU palette constants should be valid hex colour strings."""

    def test_all_constants_are_hex(self):
        for name, val in [
            ("DTU_RED", DTU_RED), ("DTU_BLUE", DTU_BLUE),
            ("DTU_NAVY", DTU_NAVY), ("DTU_BRIGHT_GREEN", DTU_BRIGHT_GREEN),
            ("DTU_ORANGE", DTU_ORANGE), ("DTU_YELLOW", DTU_YELLOW),
            ("DTU_GREY", DTU_GREY), ("DTU_DARK_GREY", DTU_DARK_GREY),
            ("DTU_PURPLE", DTU_PURPLE),
        ]:
            assert isinstance(val, str), f"{name} is not a string"
            assert val.startswith("#"), f"{name} doesn't start with #"
            assert len(val) == 7, f"{name} length is {len(val)}, expected 7"

    def test_get_dtu_palette_returns_all_keys(self):
        p = get_dtu_palette()
        expected_keys = {"red", "blue", "navy", "green", "orange", "yellow",
                         "grey", "dark_grey", "purple"}
        assert set(p.keys()) == expected_keys

    def test_all_dtu_constants_in_module_all(self):
        """All DTU colour constants should be exported via __all__."""
        import master_thesis.synthetic_plots as sp
        expected_constants = [
            "DTU_RED", "DTU_BLUE", "DTU_NAVY", "DTU_BRIGHT_GREEN",
            "DTU_ORANGE", "DTU_YELLOW", "DTU_GREY", "DTU_DARK_GREY", "DTU_PURPLE",
        ]
        for name in expected_constants:
            assert name in sp.__all__, f"{name} missing from __all__"

    def test_method_colors_use_dtu_palette(self):
        """All method colours should come from the DTU palette."""
        dtu_values = set(get_dtu_palette().values())
        for method, color in _METHOD_COLORS.items():
            assert color in dtu_values, (
                f"_METHOD_COLORS['{method}'] = '{color}' is not a DTU palette colour"
            )

    def test_aug_colors_use_dtu_palette(self):
        """All augmentation colours should come from the DTU palette."""
        dtu_values = set(get_dtu_palette().values())
        for aug, color in _AUG_COLORS.items():
            assert color in dtu_values, (
                f"_AUG_COLORS['{aug}'] = '{color}' is not a DTU palette colour"
            )

    def test_dtu_cycle_uses_dtu_palette(self):
        """The cycling palette should be a subset of DTU colours."""
        dtu_values = set(get_dtu_palette().values())
        for i, color in enumerate(_DTU_CYCLE):
            assert color in dtu_values, (
                f"_DTU_CYCLE[{i}] = '{color}' is not a DTU palette colour"
            )


# ---------------------------------------------------------------------------
# apply_dtu_style
# ---------------------------------------------------------------------------

class TestApplyDtuStyle:
    def test_sets_rcparams(self):
        import matplotlib.pyplot as plt
        apply_dtu_style()
        assert plt.rcParams["font.size"] == 11
        assert plt.rcParams["figure.dpi"] == 150
        assert plt.rcParams["axes.grid"] is True


# ---------------------------------------------------------------------------
# load_results / validation
# ---------------------------------------------------------------------------

class TestLoadResults:
    def test_accepts_valid_df(self, fake_results):
        df = load_results(fake_results)
        assert len(df) == len(fake_results)

    def test_raises_on_missing_columns(self):
        bad_df = pd.DataFrame({"x": [1, 2]})
        with pytest.raises(ValueError, match="Missing required columns"):
            load_results(bad_df)


# ---------------------------------------------------------------------------
# aggregate_by
# ---------------------------------------------------------------------------

class TestAggregateBy:
    def test_output_has_expected_columns(self, fake_results):
        agg = aggregate_by(
            fake_results,
            group_cols=("method",),
            metric_cols=("auroc",),
        )
        assert "method" in agg.columns
        assert "auroc_mean" in agg.columns
        assert "auroc_sem" in agg.columns
        assert "auroc_n" in agg.columns
        assert "n_episodes" in agg.columns

    def test_correct_group_count(self, fake_results):
        agg = aggregate_by(
            fake_results,
            group_cols=("method",),
            metric_cols=("auroc",),
        )
        expected_methods = fake_results["method"].nunique()
        assert len(agg) == expected_methods


# ---------------------------------------------------------------------------
# build_summary_table
# ---------------------------------------------------------------------------

class TestBuildSummaryTable:
    def test_contains_group_cols(self, fake_results):
        table = build_summary_table(fake_results)
        for col in GROUP_COLS_BASE:
            assert col in table.columns, f"Missing group column: {col}"

    def test_contains_metric_means(self, fake_results):
        table = build_summary_table(fake_results)
        for metric in ("auroc", "ap", "ndcg_at_10", "ece"):
            assert f"{metric}_mean" in table.columns, f"Missing {metric}_mean"

    def test_contains_collapse_rate(self, fake_results):
        table = build_summary_table(fake_results)
        assert "collapse_rate" in table.columns

    def test_extra_group_cols(self, fake_results):
        table = build_summary_table(fake_results, extra_group_cols=["init_name"])
        assert "init_name" in table.columns


# ---------------------------------------------------------------------------
# summarize_stage2b_results
# ---------------------------------------------------------------------------

class TestSummarizeStage2bResults:
    def test_returns_dataframe(self, fake_results):
        summary = summarize_stage2b_results(fake_results)
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) > 0

    def test_contains_expected_group_cols(self, fake_results):
        summary = summarize_stage2b_results(fake_results)
        for col in ("department", "real_k", "method", "init_name"):
            assert col in summary.columns, f"Missing group column: {col}"

    def test_contains_metric_aggregates(self, fake_results):
        summary = summarize_stage2b_results(fake_results, metrics=["auroc", "ap"])
        assert "auroc_mean" in summary.columns
        assert "auroc_std" in summary.columns
        assert "auroc_count" in summary.columns
        assert "ap_mean" in summary.columns

    def test_count_column(self, fake_results):
        summary = summarize_stage2b_results(fake_results)
        assert "count" in summary.columns
        assert (summary["count"] > 0).all()

    def test_handles_missing_metrics_gracefully(self, fake_results):
        """If a requested metric doesn't exist, it's silently skipped."""
        summary = summarize_stage2b_results(fake_results, metrics=["nonexistent_metric"])
        assert isinstance(summary, pd.DataFrame)
        assert "nonexistent_metric_mean" not in summary.columns


# ---------------------------------------------------------------------------
# Plotting functions — smoke tests on tiny data
# ---------------------------------------------------------------------------

class TestPlotAurocVsRealK:
    def test_runs_without_error(self, fake_results):
        import matplotlib.pyplot as plt
        fig = plot_auroc_vs_real_k(fake_results)
        assert fig is not None
        plt.close(fig)

    def test_with_filters(self, fake_results):
        import matplotlib.pyplot as plt
        fig = plot_auroc_vs_real_k(
            fake_results, method="finetune", department="dept_A",
        )
        assert fig is not None
        plt.close(fig)


class TestPlotAurocVsSynProp:
    def test_runs_without_error(self, fake_results):
        import matplotlib.pyplot as plt
        fig = plot_auroc_vs_syn_prop(fake_results)
        assert fig is not None
        plt.close(fig)


class TestPlotRealismVsSynProp:
    def test_runs_with_synthetic_rows(self, fake_results):
        import matplotlib.pyplot as plt
        fig = plot_realism_vs_syn_prop(fake_results)
        assert fig is not None
        plt.close(fig)

    def test_runs_with_no_synthetic_rows(self):
        """When all n_synthetic_support_rows == 0, shows empty message."""
        import matplotlib.pyplot as plt
        df = _make_fake_results(20)
        df["n_synthetic_support_rows"] = 0
        fig = plot_realism_vs_syn_prop(df)
        assert fig is not None
        plt.close(fig)


class TestPlotStage2bAdaptationCurve:
    def test_runs_without_error(self, fake_results):
        import matplotlib.pyplot as plt
        fig = plot_stage2b_adaptation_curve(fake_results)
        assert fig is not None
        plt.close(fig)

    def test_different_metric(self, fake_results):
        import matplotlib.pyplot as plt
        fig = plot_stage2b_adaptation_curve(fake_results, metric="ap")
        assert fig is not None
        plt.close(fig)

    def test_hue_by_method(self, fake_results):
        import matplotlib.pyplot as plt
        fig = plot_stage2b_adaptation_curve(
            fake_results, hue_col="method", group_col="department",
        )
        assert fig is not None
        plt.close(fig)

    def test_empty_data(self):
        import matplotlib.pyplot as plt
        df = _make_fake_results(10)
        df["auroc"] = np.nan
        fig = plot_stage2b_adaptation_curve(df, metric="auroc")
        assert fig is not None
        plt.close(fig)

    def test_fallback_to_gen_method(self):
        """When augmentation_method_actual is missing, falls back to gen_method."""
        import matplotlib.pyplot as plt
        df = _make_fake_results(20)
        df = df.drop(columns=["augmentation_method_actual"])
        fig = plot_stage2b_adaptation_curve(df)
        assert fig is not None
        plt.close(fig)


class TestPlotStage2bShotMatrix:
    def test_runs_without_error(self, fake_results):
        import matplotlib.pyplot as plt
        fig = plot_stage2b_shot_matrix(fake_results)
        assert fig is not None
        plt.close(fig)

    def test_with_filters(self, fake_results):
        import matplotlib.pyplot as plt
        fig = plot_stage2b_shot_matrix(
            fake_results, department="dept_A", method="finetune",
        )
        assert fig is not None
        plt.close(fig)

    def test_empty_data(self):
        import matplotlib.pyplot as plt
        df = _make_fake_results(10)
        df["auroc"] = np.nan
        fig = plot_stage2b_shot_matrix(df, metric="auroc")
        assert fig is not None
        plt.close(fig)

    def test_different_metric(self, fake_results):
        import matplotlib.pyplot as plt
        fig = plot_stage2b_shot_matrix(fake_results, metric="ap")
        assert fig is not None
        plt.close(fig)
