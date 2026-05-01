"""
run_stage2.py — Sweep runner for the Stage 2 pipeline.

Responsibilities:
- load stage2_config.yaml
- resolve a preset or the full experiment grid
- execute Stage 2 experiments systematically
- skip completed runs when configured
- save/update a master experiment summary table

This script is the CLI companion to the shared Stage 2 backend in
src/master_thesis/stage2.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import yaml

from master_thesis.config import MODELS_STAGE2
from master_thesis.stage2 import (
    build_experiment_preview_df,
    load_stage2_config,
    resolve_experiment_grid,
    run_stage2_pipeline,
)


def _load_summary_json(summary_path: Path) -> Dict[str, Any]:
    """Load a saved stage2_result_summary.json file."""
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _collect_summary_row(
    experiment_id: str,
    resolved_config: Dict[str, Any],
    pipeline_output: Optional[Dict[str, Any]] = None,
    summary_json_path: Optional[Path] = None,
    was_skipped: bool = False,
) -> Dict[str, Any]:
    """
    Build a standardized summary row for the master experiment registry.

    Priority:
    1. pipeline_output["summary"] when a fresh run occurred
    2. stage2_result_summary.json when loading an existing completed run
    """
    if pipeline_output is not None and "summary" in pipeline_output:
        summary = dict(pipeline_output["summary"])
    elif summary_json_path is not None and summary_json_path.exists():
        summary = dict(_load_summary_json(summary_json_path))
    else:
        summary = {
            "experiment_id": experiment_id,
            "method": resolved_config.get("method"),
            "init_name": resolved_config.get("stage1_init", {}).get("init_name"),
            "target_department": resolved_config.get("task_config", {}).get("target_department"),
            "n_support_pos": resolved_config.get("support_config", {}).get("n_support_pos"),
            "n_support_neg": resolved_config.get("support_config", {}).get("n_support_neg"),
            "inner_steps": resolved_config.get("meta_config", {}).get("inner_steps"),
            "status": "unknown",
            "n_metric_rows": 0,
            "n_predictions": 0,
        }

    summary["experiment_id"] = experiment_id
    summary["method"] = resolved_config.get("method")
    summary["init_name"] = resolved_config.get("stage1_init", {}).get("init_name")
    summary["target_department"] = resolved_config.get("task_config", {}).get("target_department")
    summary["n_support_pos"] = resolved_config.get("support_config", {}).get("n_support_pos")
    summary["n_support_neg"] = resolved_config.get("support_config", {}).get("n_support_neg")
    summary["inner_steps"] = resolved_config.get("meta_config", {}).get("inner_steps")
    summary["inner_lr"] = resolved_config.get("meta_config", {}).get("inner_lr")
    summary["meta_batch_size"] = resolved_config.get("meta_config", {}).get("meta_batch_size")
    summary["was_skipped"] = bool(was_skipped)
    return summary


def run_experiments(
    experiment_list: Sequence[Tuple[str, Dict[str, Any]]],
    skip_if_exists: bool = True,
    force_rerun: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run a collection of resolved Stage 2 experiments.

    Parameters
    ----------
    experiment_list
        Sequence of (experiment_id, resolved_config) pairs.
    skip_if_exists
        If True, skip runs whose summary artifact already exists.
    force_rerun
        If True, ignore existing artifacts and rerun experiments.
    verbose
        If True, print compact execution progress.

    Returns
    -------
    pd.DataFrame
        Master experiment summary with one row per experiment.
    """
    experiment_root = MODELS_STAGE2 / "experiments"
    experiment_root.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []

    for experiment_id, resolved_config in experiment_list:
        output_dir = experiment_root / experiment_id
        summary_json_path = output_dir / "stage2_result_summary.json"

        should_skip = skip_if_exists and (not force_rerun) and summary_json_path.exists()

        if should_skip:
            if verbose:
                print(f"Skipping existing experiment: {experiment_id}")
            summary_rows.append(
                _collect_summary_row(
                    experiment_id=experiment_id,
                    resolved_config=resolved_config,
                    summary_json_path=summary_json_path,
                    was_skipped=True,
                )
            )
            continue

        if verbose:
            print(f"Running experiment: {experiment_id}")

        try:
            result = run_stage2_pipeline(
                config=resolved_config,
                experiment_id=experiment_id,
            )
            summary_rows.append(
                _collect_summary_row(
                    experiment_id=experiment_id,
                    resolved_config=resolved_config,
                    pipeline_output=result,
                    was_skipped=False,
                )
            )
        except Exception as exc:
            failure_row = {
                "experiment_id": experiment_id,
                "method": resolved_config.get("method"),
                "init_name": resolved_config.get("stage1_init", {}).get("init_name"),
                "target_department": resolved_config.get("task_config", {}).get("target_department"),
                "n_support_pos": resolved_config.get("support_config", {}).get("n_support_pos"),
                "n_support_neg": resolved_config.get("support_config", {}).get("n_support_neg"),
                "inner_steps": resolved_config.get("meta_config", {}).get("inner_steps"),
                "status": "failed",
                "error": str(exc),
                "was_skipped": False,
            }
            summary_rows.append(failure_row)
            if verbose:
                print(f"Experiment failed: {experiment_id}")
                print(f"  Error: {exc}")

    summary_df = pd.DataFrame(summary_rows)

    # Merge with existing registry so prior experiment groups are preserved.
    # Rows whose experiment_id appears in the new batch are replaced; all
    # other existing rows are kept unchanged.
    summary_path = experiment_root / "experiment_summary.csv"
    if summary_path.exists():
        try:
            existing_df = pd.read_csv(summary_path)
            new_ids = set(summary_df["experiment_id"].tolist()) if (
                not summary_df.empty and "experiment_id" in summary_df.columns
            ) else set()
            existing_kept = existing_df[~existing_df["experiment_id"].isin(new_ids)]
            summary_df = pd.concat([existing_kept, summary_df], ignore_index=True)
        except Exception:
            pass  # If loading fails, just write the new rows

    if not summary_df.empty:
        sort_cols = [c for c in ["method", "init_name", "n_support_pos", "n_support_neg", "inner_steps", "target_department"] if c in summary_df.columns]
        if sort_cols:
            summary_df = summary_df.sort_values(sort_cols).reset_index(drop=True)

    summary_df.to_csv(summary_path, index=False)

    if verbose:
        print("")
        print("Saved master experiment summary to:")
        print(summary_path)
        print(f"Total rows in registry: {len(summary_df)}")

    return summary_df


def main(
    config_path: str,
    preset: Optional[str] = None,
    preview_only: bool = False,
    force_rerun: bool = False,
) -> pd.DataFrame:
    """CLI entrypoint for Stage 2 experiment execution."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    full_config = load_stage2_config(config_path)
    experiment_list = resolve_experiment_grid(
        full_config=full_config,
        preset_name=preset,
        overrides=None,
    )

    preview_df = build_experiment_preview_df(experiment_list)

    print("Loaded Stage 2 configuration:")
    print(config_path)
    print("")
    print(f"Resolved experiments: {len(preview_df)}")
    if not preview_df.empty:
        print(preview_df)

    if preview_only:
        return preview_df

    normalized_base = full_config.get("base_config", full_config)
    output_cfg = normalized_base.get("output", {})
    skip_if_exists = output_cfg.get("skip_if_exists", True)
    verbose = normalized_base.get("verbose", True)

    summary_df = run_experiments(
        experiment_list=experiment_list,
        skip_if_exists=skip_if_exists,
        force_rerun=force_rerun or output_cfg.get("force_rerun", False),
        verbose=verbose,
    )

    print("")
    print("Stage 2 execution complete.")
    print(f"Successful / loaded rows: {len(summary_df)}")
    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stage 2 experiment sweeps")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/stage2_config.yaml",
        help="Path to Stage 2 config file",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Optional preset name from stage2_config.yaml",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Resolve and preview experiments without running them",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Rerun experiments even if prior artifacts exist",
    )

    args = parser.parse_args()

    main(
        config_path=args.config,
        preset=args.preset,
        preview_only=args.preview_only,
        force_rerun=args.force_rerun,
    )
