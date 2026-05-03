"""
stage2.py — Orchestration utilities for Stage 2 department adaptation.

Responsibilities:
- load and normalize Stage 2 configuration
- resolve experiment grids and experiment identifiers
- load the Stage 2 dataset and Stage 1 artifacts
- build contract-aware department tasks
- dispatch Stage 2 methods (zero-shot / fine-tune / ANIL / FOMAML / MAML)
- save Stage 2 artifacts and summaries

This module is the shared Stage 2 backend for both CLI execution and
notebook-driven execution.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import joblib
import numpy as np
import pandas as pd
import torch
import yaml

from master_thesis.config import DATA_PROCESSED, MODELS_STAGE1, MODELS_STAGE2, SEED
from master_thesis.episode_sampler import (
    build_department_task_table,
    filter_valid_departments,
    make_logistics_meta_test_split,
    make_target_meta_test_split,
    summarize_department_tasks,
    sample_episode_with_synthetic_support,
)
from master_thesis.mlp import TabularMLP, set_seed

from master_thesis.anil import meta_train_anil, evaluate_anil_on_target_episodes
from master_thesis.FOMAML import meta_train_fomaml, evaluate_fomaml_on_target_episodes
from master_thesis.maml import meta_train_maml, evaluate_maml_on_target_episodes
from master_thesis.meta_common import evaluate_zero_shot_on_target_episodes


# ---------------------------------------------------------------------------
# Synthetic augmentation: defaults and per-episode result-row schema
# ---------------------------------------------------------------------------
# Default augmentation configuration when the user/preset does not provide
# anything explicit.  The "no-op" baseline preserves the historical behavior
# (no synthetic rows ever added).  All values are validated downstream by
# `sample_episode_with_synthetic_support` and `augment_support`.
DEFAULT_SYNTHETIC_CONFIG: Dict[str, Any] = {
    "augmentation_method": "none",   # one of: "none", "smote_nc"
    "synthetic_proportion": 0.0,     # in [0, 1]; 0.0 ⇒ no synthetic rows
    "target_per_class": None,        # optional override that pins each class
    "k_neighbors": 5,                # SMOTE-NC neighborhood size
    "categorical_cols": None,        # resolved at episode-time when None
    "augmentation_seed_offset": 10_007,
}

VALID_AUGMENTATION_METHODS: Tuple[str, ...] = ("none", "smote_nc")

# Per-episode results-row schema for the synthetic-augmentation experiment.
# This is the canonical column ordering for the long-form results CSV that
# notebook 16.0 writes (one row per (department, real_k, syn_prop, gen_method,
# init, method, seed, episode_idx)).  Centralizing it here keeps the runner
# notebook, downstream plot code, and any schema validators in lock-step.
EPISODIC_RESULT_ROW_SCHEMA: Tuple[str, ...] = (
    # Identity / grid coordinates
    "experiment_id",
    "department",
    "init_name",
    "method",
    "real_k",                # n_support_pos == n_support_neg in the symmetric grid
    "n_support_pos",
    "n_support_neg",
    "syn_prop",              # synthetic_proportion that was requested
    "gen_method",            # augmentation_method actually used
    "target_per_class",
    "seed",
    "episode_idx",
    # Episode-level support/query sizes (post-augmentation)
    "n_real_support_rows",
    "n_real_support_contracts",
    "n_synthetic_support_rows",
    "n_query_rows",
    # Predictive performance on the (real-only) query set
    "auroc",
    "ap",
    "ndcg_at_10",
    "ece",
    "pred_mean",
    "pred_std",
    "collapsed",
    # Generation-side diagnostics
    "gen_fallback_used",
    "dist_to_nearest_real_mean",
    "dist_to_nearest_real_p95",
    "ks_max",
    "ks_median",
    "cat_mode_preservation_mean",
    "cat_mode_preservation_min",
)


@dataclass
class Stage2Paths:
    task_table_path: Path
    stage1_model_path: Optional[Path]
    stage1_preprocessor_path: Optional[Path]
    output_dir: Path
    # Diagnostic fields populated by ``resolve_stage2_paths`` so callers (and
    # the pre-flight validator) can audit *exactly* where artifacts came from.
    init_scope: str = "unknown"          # "global" | "target_aware" | "none"
    init_source_dir: Optional[Path] = None  # The Stage 1 artifact directory used
    target_department: Optional[str] = None


# ---------------------------------------------------------------------------
# Stage 1 artifact layout
# ---------------------------------------------------------------------------
# After the Stage 1 restructuring, Stage 1 artifacts live under:
#
#   models/stage_1/global/A_weak_only/...
#   models/stage_1/{target_department}/B_gold_only/...
#   models/stage_1/{target_department}/C_hybrid/...
#   models/stage_1/{target_department}/D_hybrid_unweighted/...
#
# The legacy flat layout (models/stage_1/{init_name}/...) MUST NOT be used by
# Stage 2 anymore — leaving it active caused target-specific runs to silently
# load global B/C/D weights that had not seen the held-out department excluded
# from meta-training.  The constants below make the routing rule explicit and
# easy to audit.
GLOBAL_INIT_NAMES: Tuple[str, ...] = ("A_weak_only",)
TARGET_AWARE_INIT_NAMES: Tuple[str, ...] = (
    "B_gold_only",
    "C_hybrid",
    "D_hybrid_unweighted",
)
RANDOM_INIT_NAME: str = "random"
SUPPORTED_INIT_NAMES: Tuple[str, ...] = (
    GLOBAL_INIT_NAMES + TARGET_AWARE_INIT_NAMES + (RANDOM_INIT_NAME,)
)


def _safe_target_dirname(target_department: str) -> str:
    """Translate a target department label into a filesystem-safe directory.

    Stage 1 artifact directories are written using the same string the user
    supplies in ``task_config.target_department``.  Stage 2 must therefore use
    a byte-identical key to look them up; we intentionally do NOT sanitise
    spaces or special characters away.  This helper exists to centralise the
    convention so that any future sanitisation lives in one place.
    """
    return str(target_department)


def stage1_artifact_dir(
    init_name: str,
    target_department: Optional[str] = None,
    stage1_root: Optional[Path] = None,
) -> Path:
    """Return the canonical Stage 1 artifact directory for ``(init, target)``.

    The routing rule is:

    * ``A_weak_only``                       → ``stage_1/global/A_weak_only``
    * ``B_gold_only`` / ``C_hybrid`` /
      ``D_hybrid_unweighted``               → ``stage_1/{target}/{init}``
    * ``random``                            → not applicable; raises ValueError

    No fallback to the legacy flat layout is performed.  Callers that need to
    handle "missing artifact" gracefully should call
    :func:`discover_stage1_artifacts` instead.
    """
    if stage1_root is None:
        stage1_root = MODELS_STAGE1
    if init_name in GLOBAL_INIT_NAMES:
        return stage1_root / "global" / init_name
    if init_name in TARGET_AWARE_INIT_NAMES:
        if not target_department:
            raise ValueError(
                f"Init '{init_name}' is target-aware and requires a "
                f"target_department; got None."
            )
        return stage1_root / _safe_target_dirname(target_department) / init_name
    if init_name == RANDOM_INIT_NAME:
        raise ValueError(
            "Init 'random' has no Stage 1 artifact directory. "
            "Resolve its preprocessor via stage1_artifact_dir(random_preprocessor_source=...) instead."
        )
    raise ValueError(
        f"Unknown Stage 1 init '{init_name}'. "
        f"Supported: {SUPPORTED_INIT_NAMES}"
    )


def discover_stage1_artifacts(
    stage1_root: Optional[Path] = None,
    model_filename: str = "mlp_pretrained.pt",
    preprocessor_filename: str = "mlp_pretrained_preprocessor.joblib",
) -> pd.DataFrame:
    """Scan the Stage 1 artifact tree under the new target-aware layout.

    Returns one row per ``(target_department, init_name)`` cell that exists on
    disk, with explicit columns for the model/preprocessor presence.  This is
    the building block for pre-flight validation: callers can filter by
    ``has_model`` and ``has_preprocessor`` to find runnable Stage 2 cells, or
    simply ``print(df)`` for an audit of what is currently available.

    Columns
    -------
    init_name : str
        One of A_weak_only, B_gold_only, C_hybrid, D_hybrid_unweighted.
    target_department : str | None
        Department label (None for the global ``A_weak_only`` artifact).
    scope : str
        ``"global"`` for ``A_weak_only`` and ``"target_aware"`` otherwise.
    artifact_dir : str
        Absolute path to the artifact directory.
    has_model : bool
    has_preprocessor : bool
    is_runnable : bool
        Convenience: ``has_model and has_preprocessor``.

    Notes
    -----
    Random init is not enumerated here because it does not produce Stage 1
    weights.  Use :func:`resolve_stage2_paths` with
    ``stage1_init.init_name == "random"`` to surface its preprocessor path.
    """
    if stage1_root is None:
        stage1_root = MODELS_STAGE1
    rows: List[Dict[str, Any]] = []

    # Global init(s)
    global_root = stage1_root / "global"
    for init_name in GLOBAL_INIT_NAMES:
        d = global_root / init_name
        rows.append(
            {
                "init_name": init_name,
                "target_department": None,
                "scope": "global",
                "artifact_dir": str(d),
                "has_model": (d / model_filename).exists(),
                "has_preprocessor": (d / preprocessor_filename).exists(),
            }
        )

    # Target-aware inits — enumerate every direct child of stage1_root that
    # contains at least one of the target-aware sub-directories.
    skip_dirs = {"global", "experiments"}
    if stage1_root.exists():
        for target_dir in sorted(stage1_root.iterdir()):
            if not target_dir.is_dir() or target_dir.name in skip_dirs:
                continue
            # Skip legacy flat directories that look like an init_name; they
            # are NOT a valid target-aware artifact location.
            if target_dir.name in TARGET_AWARE_INIT_NAMES + GLOBAL_INIT_NAMES:
                continue
            for init_name in TARGET_AWARE_INIT_NAMES:
                d = target_dir / init_name
                if not d.exists():
                    # We still emit a row so the caller can see "expected but
                    # missing" cells instead of having to guess.
                    rows.append(
                        {
                            "init_name": init_name,
                            "target_department": target_dir.name,
                            "scope": "target_aware",
                            "artifact_dir": str(d),
                            "has_model": False,
                            "has_preprocessor": False,
                        }
                    )
                    continue
                rows.append(
                    {
                        "init_name": init_name,
                        "target_department": target_dir.name,
                        "scope": "target_aware",
                        "artifact_dir": str(d),
                        "has_model": (d / model_filename).exists(),
                        "has_preprocessor": (d / preprocessor_filename).exists(),
                    }
                )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["is_runnable"] = df["has_model"] & df["has_preprocessor"]
    else:
        df["is_runnable"] = []
    return df


def validate_experiment_grid(
    experiment_list: Sequence[Tuple[str, Dict[str, Any]]],
    stage1_root: Optional[Path] = None,
    drop_invalid: bool = False,
    verbose: bool = True,
) -> Tuple[List[Tuple[str, Dict[str, Any]]], pd.DataFrame]:
    """Pre-flight check: confirm every cell has the Stage 1 artifacts it needs.

    For each ``(experiment_id, config)`` pair, resolves the canonical Stage 1
    paths under the new layout and records whether the model and preprocessor
    exist.  Random-init cells are validated against their fallback
    preprocessor (default: ``stage_1/global/A_weak_only``).

    Parameters
    ----------
    experiment_list
        Output of :func:`resolve_experiment_grid`.
    stage1_root
        Override the Stage 1 root directory (defaults to ``MODELS_STAGE1``).
    drop_invalid
        When True, the returned ``runnable_list`` excludes cells with missing
        artifacts.  When False, the original list is returned unchanged.
    verbose
        Print a short report when invalid cells are detected.

    Returns
    -------
    runnable_list, report_df
        ``runnable_list`` is a list with the same shape as ``experiment_list``,
        optionally filtered.  ``report_df`` always covers every input cell so
        the caller can display a full audit table even after dropping.
    """
    rows: List[Dict[str, Any]] = []
    runnable: List[Tuple[str, Dict[str, Any]]] = []
    for exp_id, cfg in experiment_list:
        try:
            paths = resolve_stage2_paths(cfg, experiment_id=exp_id, stage1_root=stage1_root)
            init_name_here = cfg["stage1_init"]["init_name"]
            source_dir = paths.init_source_dir

            # --- Basic artifact presence ---
            model_present = (
                paths.stage1_model_path is not None
                and paths.stage1_model_path.exists()
            )
            preprocessor_present = (
                paths.stage1_preprocessor_path is not None
                and paths.stage1_preprocessor_path.exists()
            )

            # --- Metadata presence ---
            metadata_path = (source_dir / "metadata.json") if source_dir else None
            metadata_present = metadata_path is not None and metadata_path.exists()

            # --- feature_names.json presence (warning only, not blocking) ---
            feature_names_path = (source_dir / "feature_names.json") if source_dir else None
            feature_names_present = feature_names_path is not None and feature_names_path.exists()
            feature_names_warning = (
                None if (feature_names_present or init_name_here == RANDOM_INIT_NAME)
                else (
                    f"feature_names.json missing at '{source_dir}'. "
                    f"SHAP and interpretability code may fail. "
                    f"Re-run Stage 1 training to regenerate."
                )
            )

            # --- Architecture compatibility check ---
            arch_warning = None
            arch_compatible = True
            if init_name_here != RANDOM_INIT_NAME and metadata_present and model_present:
                try:
                    meta = load_stage1_artifact_metadata(source_dir)
                    mlp_arch = meta.get("mlp_architecture")
                    stage2_mlp_cfg = cfg.get("mlp_config", {})
                    if mlp_arch is None:
                        arch_warning = (
                            f"metadata.json at '{source_dir}' is missing "
                            f"'mlp_architecture'. Cannot verify architecture "
                            f"compatibility. Re-run Stage 1 to regenerate."
                        )
                        arch_compatible = False
                    else:
                        s2_h1 = stage2_mlp_cfg.get("hidden_dim_1")
                        s2_h2 = stage2_mlp_cfg.get("hidden_dim_2")
                        m_h1 = mlp_arch.get("hidden_dim_1")
                        m_h2 = mlp_arch.get("hidden_dim_2")
                        if s2_h1 != m_h1 or s2_h2 != m_h2:
                            arch_warning = (
                                f"Stage 2 config mlp_config ({s2_h1}/{s2_h2}) differs "
                                f"from artifact metadata ({m_h1}/{m_h2}). "
                                f"build_stage2_model_for_init() will use the artifact "
                                f"architecture ({m_h1}/{m_h2}) for safe strict=True loading."
                            )
                            # This is NOT a blocking error — the new builder handles it.
                            # arch_compatible stays True; the warning is informational.
                except Exception as exc:
                    arch_warning = f"Architecture check failed: {exc}"
                    arch_compatible = False

            row = {
                "experiment_id": exp_id,
                "init_name": init_name_here,
                "target_department": cfg["task_config"]["target_department"],
                "method": cfg["method"],
                "init_scope": paths.init_scope,
                "init_source_dir": str(source_dir) if source_dir else None,
                "model_path": str(paths.stage1_model_path) if paths.stage1_model_path else None,
                "preprocessor_path": str(paths.stage1_preprocessor_path) if paths.stage1_preprocessor_path else None,
                "model_present": model_present,
                "preprocessor_present": preprocessor_present,
                "metadata_present": metadata_present,
                "feature_names_present": feature_names_present,
                "arch_compatible": arch_compatible,
                "arch_warning": arch_warning,
                "feature_names_warning": feature_names_warning,
                "resolution_error": None,
            }
        except Exception as exc:
            row = {
                "experiment_id": exp_id,
                "init_name": cfg.get("stage1_init", {}).get("init_name"),
                "target_department": cfg.get("task_config", {}).get("target_department"),
                "method": cfg.get("method"),
                "init_scope": "error",
                "init_source_dir": None,
                "model_path": None,
                "preprocessor_path": None,
                "model_present": False,
                "preprocessor_present": False,
                "metadata_present": False,
                "feature_names_present": False,
                "arch_compatible": False,
                "arch_warning": None,
                "feature_names_warning": None,
                "resolution_error": str(exc),
            }

        # Random init: needs preprocessor only (no model weights, no metadata).
        # Pretrained init: needs model + preprocessor + metadata (for arch-safe
        #   loading via build_stage2_model_for_init).  Arch mismatch vs the
        #   stage2_config is ALLOWED because the builder reads from metadata.
        if row["init_name"] == RANDOM_INIT_NAME:
            row["is_runnable"] = (
                bool(row["preprocessor_present"])
                and row["resolution_error"] is None
            )
        else:
            row["is_runnable"] = (
                bool(row["model_present"])
                and bool(row["preprocessor_present"])
                and bool(row["metadata_present"])
                and bool(row["arch_compatible"])
                and row["resolution_error"] is None
            )

        rows.append(row)
        if row["is_runnable"]:
            runnable.append((exp_id, cfg))

        # Emit non-blocking warnings even for runnable cells.
        if verbose and row.get("arch_warning"):
            print(f"[Stage 2 pre-flight] ARCH NOTE  [{exp_id}]: {row['arch_warning']}")
        if verbose and row.get("feature_names_warning"):
            print(f"[Stage 2 pre-flight] FN WARNING [{exp_id}]: {row['feature_names_warning']}")

    report_df = pd.DataFrame(rows)
    invalid = report_df[~report_df["is_runnable"]] if not report_df.empty else report_df
    if verbose and not invalid.empty:
        print(f"\n[Stage 2 pre-flight] {len(invalid)} of {len(report_df)} cell(s) are NOT runnable:")
        cols = [
            "experiment_id", "init_name", "target_department", "method",
            "model_present", "preprocessor_present", "metadata_present",
            "arch_compatible", "resolution_error",
        ]
        print(invalid[[c for c in cols if c in invalid.columns]].to_string(index=False))

    return (runnable if drop_invalid else list(experiment_list)), report_df


def load_stage2_config(config_path: str | Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def normalize_stage2_config(config: Dict[str, Any]) -> Dict[str, Any]:
    if "base_config" in config:
        return copy.deepcopy(config["base_config"])
    return copy.deepcopy(config)


def generate_experiment_id(
    init_name: str,
    method: str,
    n_support_pos: int,
    n_support_neg: int,
    inner_steps: int,
    target_department: str,
    inner_lr: Optional[float] = None,
    synthetic_proportion: Optional[float] = None,
    augmentation_method: Optional[str] = None,
    target_effective_k: Optional[int] = None,
    gradient_clip_norm: Optional[float] = None,
) -> str:
    """Build a deterministic experiment ID.

    The synthetic-augmentation suffixes (``__synprop-…``, ``__gen-…``,
    ``__tek-…``) are appended only when explicitly provided.  This keeps legacy
    experiment directories (which were generated before synthetic augmentation
    existed) byte-identical, while letting new experiments carve out their own
    directories per ``(syn_prop, gen_method, target_effective_k)`` combination.
    """
    safe_target = str(target_department).replace(" ", "-").replace("/", "-")
    id_str = (
        f"stage2__init-{init_name}"
        f"__method-{method}"
        f"__kpos-{n_support_pos}"
        f"__kneg-{n_support_neg}"
        f"__steps-{inner_steps}"
        f"__target-{safe_target}"
    )
    if inner_lr is not None:
        # Format as e.g. __lr-0.001 — strip trailing zeros for readability.
        lr_str = f"{inner_lr:.6f}".rstrip("0").rstrip(".")
        id_str += f"__lr-{lr_str}"
    if synthetic_proportion is not None:
        # Format as e.g. __synprop-0.5 — strip trailing zeros.
        sp = f"{float(synthetic_proportion):.4f}".rstrip("0").rstrip(".")
        if sp == "" or sp == "-":
            sp = "0"
        id_str += f"__synprop-{sp}"
    if augmentation_method is not None:
        safe_gen = str(augmentation_method).replace(" ", "-").replace("/", "-")
        id_str += f"__gen-{safe_gen}"
    if target_effective_k is not None:
        id_str += f"__tek-{int(target_effective_k)}"
    if gradient_clip_norm is not None:
        clip_str = f"{gradient_clip_norm:.4f}".rstrip("0").rstrip(".")
        if clip_str == "" or clip_str == "-":
            clip_str = "0"
        id_str += f"__clip-{clip_str}"
    return id_str


def _resolve_synthetic_axes(
    experiment_grid: Dict[str, Any],
    base_config: Dict[str, Any],
) -> Tuple[List[float], List[str], bool]:
    """Pull the synthetic-augmentation axes out of an experiment_grid.

    Returns ``(synthetic_proportion_grid, augmentation_method_grid, sweep)``,
    where ``sweep`` is True iff the user actually requested a sweep (and
    therefore the ID should encode the synthetic axes).  When neither axis is
    declared in the grid we return single-element lists drawn from
    ``base_config["synthetic_config"]`` (or the global defaults), and
    ``sweep=False`` so legacy IDs stay unchanged.
    """
    synprop_raw = experiment_grid.get("synthetic_proportion_grid", None)
    method_raw = experiment_grid.get("augmentation_method_grid", None)
    sweep = (synprop_raw is not None) or (method_raw is not None)

    base_syn = (base_config.get("synthetic_config") or {}) if isinstance(
        base_config.get("synthetic_config"), dict
    ) else {}

    if synprop_raw is None:
        synprop_grid = [float(base_syn.get(
            "synthetic_proportion", DEFAULT_SYNTHETIC_CONFIG["synthetic_proportion"]
        ))]
    else:
        synprop_grid = [float(p) for p in synprop_raw]

    if method_raw is None:
        method_grid = [str(base_syn.get(
            "augmentation_method", DEFAULT_SYNTHETIC_CONFIG["augmentation_method"]
        ))]
    else:
        method_grid = [str(m) for m in method_raw]

    for m in method_grid:
        if m not in VALID_AUGMENTATION_METHODS:
            raise ValueError(
                f"Unknown augmentation_method '{m}'. "
                f"Valid options: {VALID_AUGMENTATION_METHODS}"
            )
    for p in synprop_grid:
        if not (0.0 <= p <= 1.0):
            raise ValueError(
                f"synthetic_proportion must lie in [0, 1]; got {p}"
            )

    return synprop_grid, method_grid, sweep


def _build_synthetic_block(
    base_config: Dict[str, Any],
    experiment_grid: Dict[str, Any],
    synthetic_proportion: float,
    augmentation_method: str,
    target_effective_k: Optional[int] = None,
) -> Dict[str, Any]:
    """Build the per-experiment ``synthetic_config`` block.

    Order of precedence (low → high):
      1. ``DEFAULT_SYNTHETIC_CONFIG`` module constant
      2. ``base_config["synthetic_config"]`` if present
      3. ``experiment_grid["synthetic_config"]`` if present (preset-level)
      4. The per-cell axes (synthetic_proportion, augmentation_method,
         target_effective_k)
    """
    block = copy.deepcopy(DEFAULT_SYNTHETIC_CONFIG)
    if isinstance(base_config.get("synthetic_config"), dict):
        block.update(copy.deepcopy(base_config["synthetic_config"]))
    if isinstance(experiment_grid.get("synthetic_config"), dict):
        block.update(copy.deepcopy(experiment_grid["synthetic_config"]))
    block["synthetic_proportion"] = float(synthetic_proportion)
    block["augmentation_method"] = str(augmentation_method)
    # When the augmentation is disabled we must not leak a positive proportion
    # into downstream code that might branch on either signal.  Pin both.
    if block["augmentation_method"] == "none":
        block["synthetic_proportion"] = 0.0
        block["target_per_class"] = None
    elif target_effective_k is not None:
        # target_effective_k overrides synthetic_proportion for support sizing.
        # Store as target_per_class so downstream episode sampler sees it.
        block["target_per_class"] = int(target_effective_k)
    return block


def resolve_target_departments_from_data(
    task_df: "pd.DataFrame",
    config: Dict[str, Any],
    department_col: str = "department",
    min_pos_contracts: int = 2,
    min_neg_contracts: int = 2,
) -> List[str]:
    """Return a sorted list of departments eligible to serve as meta-test targets.

    This helper is the data-aware resolution path for ``"all_eligible"`` in the
    ``target_departments`` experiment grid axis.  It queries the loaded
    ``task_df`` rather than the static YAML, so eligibility is always consistent
    with the actual data on disk.

    Eligibility criteria (conservative defaults):
    - The department must appear in ``task_df``.
    - It must have at least ``min_pos_contracts`` distinct positive-class
      contracts and ``min_neg_contracts`` distinct negative-class contracts so
      that at least one support/query split can be formed.

    Parameters
    ----------
    task_df : pd.DataFrame
        Gold-labeled task table (output of ``build_department_task_table``).
    config : dict
        Stage 2 config dict (used to read ``target_col`` and ``contract_id_col``
        from ``config["data"]``).
    department_col : str
        Column identifying the department (default ``"department"``).
    min_pos_contracts : int
        Minimum number of distinct positive-class contracts required.
    min_neg_contracts : int
        Minimum number of distinct negative-class contracts required.

    Returns
    -------
    List[str]
        Sorted list of eligible department names.  Raises ``ValueError`` if
        no departments meet the criteria.
    """
    target_col = config.get("data", {}).get("target_col", "gold_y")
    contract_id_col = config.get("data", {}).get("group_col", "contract_id")

    eligible: List[str] = []
    for dept, grp in task_df.groupby(department_col, sort=True):
        pos_contracts = grp.loc[grp[target_col] == 1, contract_id_col].nunique()
        neg_contracts = grp.loc[grp[target_col] == 0, contract_id_col].nunique()
        if pos_contracts >= min_pos_contracts and neg_contracts >= min_neg_contracts:
            eligible.append(str(dept))

    if not eligible:
        raise ValueError(
            "resolve_target_departments_from_data: no department in task_df meets "
            f"the minimum eligibility criteria (min_pos_contracts={min_pos_contracts}, "
            f"min_neg_contracts={min_neg_contracts})."
        )
    return sorted(eligible)


def resolve_experiment_grid(
    full_config: Dict[str, Any],
    preset_name: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    overrides = overrides or {}

    if "base_config" not in full_config:
        resolved = normalize_stage2_config(full_config)
        # Even in the single-experiment case, materialize a synthetic_config
        # block so downstream code can rely on it being present.
        existing_syn = resolved.get("synthetic_config") or {}
        syn_block = _build_synthetic_block(
            base_config=resolved,
            experiment_grid={},
            synthetic_proportion=float(
                existing_syn.get(
                    "synthetic_proportion",
                    DEFAULT_SYNTHETIC_CONFIG["synthetic_proportion"],
                )
            ),
            augmentation_method=str(
                existing_syn.get(
                    "augmentation_method",
                    DEFAULT_SYNTHETIC_CONFIG["augmentation_method"],
                )
            ),
        )
        resolved["synthetic_config"] = syn_block
        exp_id = generate_experiment_id(
            init_name=resolved["stage1_init"]["init_name"],
            method=resolved["method"],
            n_support_pos=resolved["support_config"]["n_support_pos"],
            n_support_neg=resolved["support_config"]["n_support_neg"],
            inner_steps=resolved["meta_config"]["inner_steps"],
            target_department=resolved["task_config"]["target_department"],
        )
        return [(exp_id, resolved)]

    base_config = copy.deepcopy(full_config["base_config"])
    presets = full_config.get("presets", {})
    experiment_grid = copy.deepcopy(full_config.get("experiment_grid", {}))

    if preset_name:
        preset_cfg = presets.get(preset_name)
        if preset_cfg is None:
            raise ValueError(f"Unknown Stage 2 preset: {preset_name}")
        experiment_grid = _deep_update(experiment_grid, preset_cfg)

    experiment_grid = _deep_update(experiment_grid, overrides)

    methods = experiment_grid.get("methods", [base_config["method"]])
    support_grid = experiment_grid.get("support_grid", [copy.deepcopy(base_config["support_config"])])
    inner_steps_grid = experiment_grid.get("inner_steps_grid", [base_config["meta_config"]["inner_steps"]])
    init_names = experiment_grid.get("init_names", [base_config["stage1_init"]["init_name"]])
    target_departments = experiment_grid.get(
        "target_departments", [base_config["task_config"]["target_department"]]
    )

    # Guard: "all_eligible" must be resolved after data loading via
    # resolve_target_departments_from_data().  If it reaches here as a bare
    # string it would be iterated character-by-character by itertools.product,
    # producing ~11 nonsensical single-letter experiment cells.
    if isinstance(target_departments, str):
        raise NotImplementedError(
            f"target_departments='{target_departments}' is a string, not a list. "
            "The special value 'all_eligible' cannot be resolved inside "
            "resolve_experiment_grid() because this function has no access to the "
            "loaded task DataFrame.  Call resolve_target_departments_from_data() "
            "after loading the data, then pass the resulting list back as the "
            "target_departments override, e.g.:\n\n"
            "    eligible = resolve_target_departments_from_data(task_df, config)\n"
            "    overrides = {'target_departments': eligible}\n"
            "    grid = resolve_experiment_grid(full_config, overrides=overrides)"
        )

    # inner_lr_grid: when present, lr is swept AND included in the experiment ID
    # so different lr values produce distinct experiment directories.
    inner_lr_grid_raw = experiment_grid.get("inner_lr_grid", None)
    sweep_lr = inner_lr_grid_raw is not None
    inner_lr_grid = (
        [float(lr) for lr in inner_lr_grid_raw]
        if sweep_lr
        else [float(base_config["meta_config"]["inner_lr"])]
    )

    # Synthetic-augmentation axes.  When the user declares either grid we
    # treat it as a sweep and embed both fields in the experiment ID; when
    # neither is declared we keep IDs byte-identical to the legacy behavior.
    synprop_grid, gen_method_grid, sweep_synth = _resolve_synthetic_axes(
        experiment_grid, base_config
    )

    # target_effective_k sweep axis.  When present in the preset, each non-null
    # value pins target_per_class for augmented cells and gets its own __tek-N
    # suffix in the experiment ID (preventing folder collision with the
    # synthetic_proportion-only runs of the same cell).  A value of null in the
    # grid means "use synthetic_proportion instead", i.e. the default behaviour.
    tek_raw = experiment_grid.get("target_effective_k_grid", None)
    if tek_raw is None:
        tek_grid: List[Optional[int]] = [None]
        sweep_tek = False
    else:
        tek_grid = [int(v) if v is not None else None for v in tek_raw]
        # Only embed __tek-N in IDs when at least one value is non-null.
        sweep_tek = any(v is not None for v in tek_grid)
        if sweep_tek:
            sweep_synth = True  # non-null tek implies an augmentation sweep

    preset_meta_overrides = {}
    for key in ["meta_iterations", "n_repeats", "inner_lr", "outer_lr", "meta_batch_size", "target_inner_steps", "inner_grad_clip"]:
        if key in experiment_grid:
            preset_meta_overrides[key] = experiment_grid[key]

    # Preset-level task_config overrides (flat keys → nested into task_config).
    preset_task_overrides = {}
    for key in ["require_both_query_classes"]:
        if key in experiment_grid:
            preset_task_overrides[key] = experiment_grid[key]

    # gradient_clip_norm_grid: when present, outer gradient clipping is swept
    # AND included in the experiment ID so different clip values produce
    # distinct experiment directories.
    clip_grid_raw = experiment_grid.get("gradient_clip_norm_grid", None)
    sweep_clip = clip_grid_raw is not None
    if sweep_clip:
        # Convert: null → None (no clipping), numbers → float
        clip_grid: List[Optional[float]] = [
            float(v) if v is not None else None for v in clip_grid_raw
        ]
    else:
        # Single-element list drawn from base_config (default 1.0 for legacy).
        clip_grid = [base_config.get("meta_config", {}).get("gradient_clip_norm", 1.0)]

    experiment_list: List[Tuple[str, Dict[str, Any]]] = []
    seen_cells: set = set()
    for (
        method,
        support_cfg,
        inner_steps,
        init_name,
        target_department,
        inner_lr,
        synthetic_proportion,
        augmentation_method,
        target_effective_k,
        gradient_clip_norm,
    ) in product(
        methods,
        support_grid,
        inner_steps_grid,
        init_names,
        target_departments,
        inner_lr_grid,
        synprop_grid,
        gen_method_grid,
        tek_grid,
        clip_grid,
    ):
        # Deduplication rule: when augmentation is disabled, the sweep over
        # synthetic_proportion is meaningless — every (syn_prop, "none") cell
        # collapses to the same run.  Canonicalize syn_prop to 0.0 in that
        # case and skip duplicates.  Symmetrically, when syn_prop=0.0 the
        # method is meaningless; canonicalize to "none".
        canonical_method = augmentation_method
        canonical_synprop = float(synthetic_proportion)
        if canonical_method == "none":
            canonical_synprop = 0.0
        elif canonical_synprop == 0.0:
            canonical_method = "none"

        # When augmentation is disabled, target_effective_k is meaningless;
        # canonicalize to None so duplicate cells are collapsed.
        canonical_tek = target_effective_k if canonical_method != "none" else None

        cell_key = (
            method,
            int(support_cfg["n_support_pos"]),
            int(support_cfg["n_support_neg"]),
            int(inner_steps),
            init_name,
            target_department,
            float(inner_lr),
            canonical_synprop,
            canonical_method,
            canonical_tek,
            gradient_clip_norm,  # None or float — both are hashable
        )
        if cell_key in seen_cells:
            continue
        seen_cells.add(cell_key)

        resolved = copy.deepcopy(base_config)
        resolved["method"] = method
        resolved["support_config"]["n_support_pos"] = int(support_cfg["n_support_pos"])
        resolved["support_config"]["n_support_neg"] = int(support_cfg["n_support_neg"])
        resolved["meta_config"]["inner_steps"] = int(inner_steps)
        resolved["stage1_init"]["init_name"] = init_name
        resolved["task_config"]["target_department"] = target_department
        resolved["meta_config"].update(copy.deepcopy(preset_meta_overrides))
        # Always apply the per-experiment inner_lr (overrides any preset_meta_overrides value)
        resolved["meta_config"]["inner_lr"] = inner_lr
        # Apply the per-experiment gradient_clip_norm.
        resolved["meta_config"]["gradient_clip_norm"] = gradient_clip_norm
        # Apply preset-level task_config overrides.
        resolved["task_config"].update(copy.deepcopy(preset_task_overrides))
        # If the preset didn't explicitly set target_inner_steps, default to
        # the experiment's inner_steps.  This prevents the common pitfall of
        # meta-training with 1 step but evaluating with the base_config's
        # default of 5 steps, which causes over-adaptation and collapse.
        if "target_inner_steps" not in preset_meta_overrides:
            resolved["meta_config"]["target_inner_steps"] = int(inner_steps)

        # Attach the resolved synthetic_config block.
        resolved["synthetic_config"] = _build_synthetic_block(
            base_config=base_config,
            experiment_grid=experiment_grid,
            synthetic_proportion=canonical_synprop,
            augmentation_method=canonical_method,
            target_effective_k=canonical_tek,
        )

        exp_id = generate_experiment_id(
            init_name=init_name,
            method=method,
            n_support_pos=resolved["support_config"]["n_support_pos"],
            n_support_neg=resolved["support_config"]["n_support_neg"],
            inner_steps=resolved["meta_config"]["inner_steps"],
            target_department=target_department,
            inner_lr=inner_lr if sweep_lr else None,
            synthetic_proportion=canonical_synprop if sweep_synth else None,
            augmentation_method=canonical_method if sweep_synth else None,
            target_effective_k=canonical_tek if sweep_tek else None,
            gradient_clip_norm=gradient_clip_norm if sweep_clip else None,
        )
        experiment_list.append((exp_id, resolved))

    exp_ids = [exp_id for exp_id, _ in experiment_list]
    if len(exp_ids) != len(set(exp_ids)):
        raise ValueError("Duplicate Stage 2 experiment IDs were generated.")

    return experiment_list


def build_experiment_preview_df(
    experiment_list: Sequence[Tuple[str, Dict[str, Any]]]
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for experiment_id, config in experiment_list:
        syn_cfg = config.get("synthetic_config") or {}
        rows.append(
            {
                "experiment_id": experiment_id,
                "method": config["method"],
                "n_support_pos": config["support_config"]["n_support_pos"],
                "n_support_neg": config["support_config"]["n_support_neg"],
                "inner_steps": config["meta_config"]["inner_steps"],
                "init_name": config["stage1_init"]["init_name"],
                "target_department": config["task_config"]["target_department"],
                "data_filename": config["data"]["data_filename"],
                "synthetic_proportion": syn_cfg.get("synthetic_proportion", 0.0),
                "augmentation_method": syn_cfg.get("augmentation_method", "none"),
                "target_per_class": syn_cfg.get("target_per_class"),
                "k_neighbors": syn_cfg.get(
                    "k_neighbors", DEFAULT_SYNTHETIC_CONFIG["k_neighbors"]
                ),
                "gradient_clip_norm": config["meta_config"].get("gradient_clip_norm"),
                "inner_grad_clip": config["meta_config"].get("inner_grad_clip"),
            }
        )
    return pd.DataFrame(rows)


def _resolve_random_preprocessor_path(
    config: Dict[str, Any],
    stage1_root: Path,
) -> Tuple[Path, Path]:
    """Resolve the preprocessor used for random-init experiments.

    Returns ``(preprocessor_path, source_dir)``.

    Random init has no pretrained weights, but it still needs a preprocessor
    so that the feature space and input dimensionality match the pretrained
    baselines.  Under the new layout this preprocessor lives at
    ``stage_1/global/A_weak_only/`` by default.  The source can be overridden
    by ``stage1_init.random_preprocessor_source``; the value may be either:

    * ``"A_weak_only"`` (resolves to ``stage_1/global/A_weak_only``)
    * a target-aware init name (``B_gold_only`` / ``C_hybrid`` /
      ``D_hybrid_unweighted``) — in which case the target_department from
      ``task_config`` is used to locate the matching folder
    """
    init_cfg = config["stage1_init"]
    preprocessor_filename = init_cfg["preprocessor_filename"]
    source = init_cfg.get("random_preprocessor_source", "A_weak_only")
    target_department = config["task_config"].get("target_department")

    if source in GLOBAL_INIT_NAMES:
        source_dir = stage1_root / "global" / source
    elif source in TARGET_AWARE_INIT_NAMES:
        if not target_department:
            raise ValueError(
                f"random_preprocessor_source='{source}' is target-aware but no "
                f"target_department was provided in task_config."
            )
        source_dir = stage1_root / _safe_target_dirname(target_department) / source
    else:
        raise ValueError(
            f"Unknown random_preprocessor_source='{source}'. "
            f"Supported: {GLOBAL_INIT_NAMES + TARGET_AWARE_INIT_NAMES}"
        )

    return source_dir / preprocessor_filename, source_dir


def resolve_stage2_paths(
    config: Dict[str, Any],
    experiment_id: Optional[str] = None,
    stage1_root: Optional[Path] = None,
) -> Stage2Paths:
    """Resolve every filesystem path Stage 2 needs for one experiment cell.

    Routing rule (new Stage 1 layout, target-aware):

    * ``A_weak_only``                               → ``stage_1/global/A_weak_only/``
    * ``B_gold_only`` / ``C_hybrid`` /
      ``D_hybrid_unweighted``                       → ``stage_1/{target}/{init}/``
    * ``random``                                    → no model file; preprocessor
      drawn from ``stage_1/global/A_weak_only/`` (or whatever
      ``random_preprocessor_source`` points to)

    The legacy flat layout (``stage_1/{init}/...``) is intentionally ignored.
    Missing artifacts are NOT silently substituted — downstream loaders raise a
    clean ``FileNotFoundError`` so a misconfigured run fails immediately rather
    than secretly using the wrong pretrained weights.
    """
    config = normalize_stage2_config(config)
    if stage1_root is None:
        stage1_root = MODELS_STAGE1

    task_table_path = DATA_PROCESSED / config["data"]["data_filename"]
    init_name = config["stage1_init"]["init_name"]
    method = config["method"]
    target_department = config["task_config"].get("target_department")

    if init_name not in SUPPORTED_INIT_NAMES:
        raise ValueError(
            f"Unsupported stage1_init.init_name='{init_name}'. "
            f"Supported: {SUPPORTED_INIT_NAMES}"
        )

    if init_name == RANDOM_INIT_NAME:
        # Random init: no pretrained weights, but the preprocessor must exist
        # so we still get a valid feature space.
        stage1_model_path: Optional[Path] = None
        stage1_preprocessor_path, source_dir = _resolve_random_preprocessor_path(
            config, stage1_root
        )
        init_scope = "none"
    elif init_name in GLOBAL_INIT_NAMES:
        source_dir = stage1_artifact_dir(init_name, stage1_root=stage1_root)
        stage1_model_path = source_dir / config["stage1_init"]["model_filename"]
        stage1_preprocessor_path = source_dir / config["stage1_init"]["preprocessor_filename"]
        init_scope = "global"
    else:
        # Target-aware: B_gold_only / C_hybrid / D_hybrid_unweighted.
        source_dir = stage1_artifact_dir(
            init_name,
            target_department=target_department,
            stage1_root=stage1_root,
        )
        stage1_model_path = source_dir / config["stage1_init"]["model_filename"]
        stage1_preprocessor_path = source_dir / config["stage1_init"]["preprocessor_filename"]
        init_scope = "target_aware"

    if experiment_id:
        output_dir = MODELS_STAGE2 / "experiments" / experiment_id
    else:
        output_dir = MODELS_STAGE2 / init_name / method
    output_dir.mkdir(parents=True, exist_ok=True)

    return Stage2Paths(
        task_table_path=task_table_path,
        stage1_model_path=stage1_model_path,
        stage1_preprocessor_path=stage1_preprocessor_path,
        output_dir=output_dir,
        init_scope=init_scope,
        init_source_dir=source_dir,
        target_department=target_department,
    )


def get_feature_cols(config: Dict[str, Any], preprocessor=None) -> List[str]:
    features_cfg = config["features"]
    if "columns" in features_cfg:
        feature_cols = features_cfg["columns"]
        if not isinstance(feature_cols, list) or len(feature_cols) == 0:
            raise ValueError("config['features']['columns'] must be a non-empty list.")
        return feature_cols

    if features_cfg.get("source") == "stage1_preprocessor":
        if preprocessor is None:
            raise ValueError("features.source='stage1_preprocessor' requires a loaded preprocessor.")
        if not hasattr(preprocessor, "feature_names_in_"):
            raise ValueError("Loaded preprocessor does not expose feature_names_in_.")
        return list(preprocessor.feature_names_in_)

    raise ValueError(
        "Could not resolve feature columns. Provide either features.columns or "
        "features.source='stage1_preprocessor'."
    )


def load_stage2_dataframe(task_table_path: Path) -> pd.DataFrame:
    if not task_table_path.exists():
        raise FileNotFoundError(f"Stage 2 dataset not found: {task_table_path}")
    return pd.read_csv(task_table_path, low_memory=False)


def load_stage1_preprocessor(preprocessor_path: Optional[Path]):
    if preprocessor_path is None:
        return None
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Stage 1 preprocessor not found: {preprocessor_path}")
    return joblib.load(preprocessor_path)


def load_stage2_raw_data(config: Dict[str, Any], experiment_id: Optional[str] = None) -> Dict[str, Any]:
    config = normalize_stage2_config(config)
    paths = resolve_stage2_paths(config, experiment_id=experiment_id)
    df = load_stage2_dataframe(paths.task_table_path)
    preprocessor = load_stage1_preprocessor(paths.stage1_preprocessor_path)
    feature_cols = get_feature_cols(config, preprocessor=preprocessor)
    return {"paths": paths, "df": df, "preprocessor": preprocessor, "feature_cols": feature_cols}


def build_stage2_model_from_config(
    input_dim: int,
    config: Dict[str, Any],
    device: Optional[torch.device] = None,
) -> TabularMLP:
    """Build a TabularMLP from stage2_config.yaml's mlp_config section.

    .. warning::
        This function is kept for backward compatibility and for random-init
        experiments.  For all pretrained Stage 1 inits (A / B / C / D) you
        MUST use :func:`build_stage2_model_for_init` instead, which reads the
        architecture from the artifact's ``metadata.json`` and guarantees that
        ``strict=True`` weight loading will succeed.
    """
    mlp_cfg = config["mlp_config"]
    model = TabularMLP(
        input_dim=input_dim,
        hidden_dim_1=mlp_cfg["hidden_dim_1"],
        hidden_dim_2=mlp_cfg["hidden_dim_2"],
        dropout=mlp_cfg["dropout"],
    )
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)


# ---------------------------------------------------------------------------
# Architecture-safe model building for Stage 2
# ---------------------------------------------------------------------------

def load_stage1_artifact_metadata(artifact_dir: Path) -> Dict[str, Any]:
    """Load and return the ``metadata.json`` from a Stage 1 artifact directory.

    Parameters
    ----------
    artifact_dir : Path
        The canonical Stage 1 condition directory, e.g.
        ``models/stage_1/global/A_weak_only/``.

    Raises
    ------
    FileNotFoundError
        If ``metadata.json`` does not exist under ``artifact_dir``.
    """
    metadata_path = artifact_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Stage 1 metadata not found: {metadata_path}\n"
            f"Re-run Stage 1 training (run_stage1.py or notebook 16) so that "
            f"metadata.json is saved alongside the model weights."
        )
    with open(metadata_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def infer_arch_from_state_dict(state_dict: Dict[str, Any]) -> Dict[str, int]:
    """Infer TabularMLP architecture from a loaded state_dict.

    The expected key layout for :class:`TabularMLP` is::

        net.0.weight  shape (h1, input_dim)   — first Linear
        net.4.weight  shape (h2, h1)           — second Linear
        net.8.weight  shape (1,  h2)           — output Linear

    Parameters
    ----------
    state_dict : dict
        PyTorch state dict (e.g. from ``torch.load(path, weights_only=True)``).

    Returns
    -------
    dict with keys ``input_dim``, ``hidden_dim_1``, ``hidden_dim_2``.

    Raises
    ------
    KeyError
        If the expected weight keys are absent (wrong model class or layout).
    """
    try:
        w0 = state_dict["net.0.weight"]  # (h1, input_dim)
        w4 = state_dict["net.4.weight"]  # (h2, h1)
    except KeyError as exc:
        raise KeyError(
            f"Cannot infer architecture from state_dict — expected keys "
            f"'net.0.weight' and 'net.4.weight', got: {list(state_dict.keys())}"
        ) from exc
    return {
        "input_dim": w0.shape[1],
        "hidden_dim_1": w0.shape[0],
        "hidden_dim_2": w4.shape[0],
    }


def validate_artifact_architecture(
    artifact_dir: Path,
    model_filename: str = "mlp_pretrained.pt",
) -> Dict[str, Any]:
    """Cross-check ``metadata.json`` against the actual ``.pt`` tensor shapes.

    Returns a dict with keys ``metadata_arch``, ``actual_arch``, ``match``,
    and ``warning`` (None if OK, else an explanation string).

    The function will not raise unless metadata is completely absent; shape
    mismatches are surfaced as warnings so the caller can decide what to do.
    """
    result: Dict[str, Any] = {
        "metadata_arch": None,
        "actual_arch": None,
        "match": None,
        "warning": None,
    }

    # 1. Load metadata architecture.
    try:
        meta = load_stage1_artifact_metadata(artifact_dir)
    except FileNotFoundError as exc:
        result["warning"] = str(exc)
        return result

    mlp_arch = meta.get("mlp_architecture")
    if mlp_arch is None:
        result["warning"] = (
            f"'mlp_architecture' key missing from metadata.json at '{artifact_dir}'. "
            f"Re-run Stage 1 training to regenerate metadata."
        )
        return result
    result["metadata_arch"] = mlp_arch

    # 2. Load state_dict and infer actual shapes.
    model_path = artifact_dir / model_filename
    if not model_path.exists():
        result["warning"] = f"Model file not found: {model_path}"
        return result

    try:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        actual = infer_arch_from_state_dict(state_dict)
        result["actual_arch"] = actual
    except Exception as exc:
        result["warning"] = f"Could not load or parse state_dict: {exc}"
        return result

    # 3. Compare.
    meta_h1 = int(mlp_arch.get("hidden_dim_1", -1))
    meta_h2 = int(mlp_arch.get("hidden_dim_2", -1))
    if meta_h1 == actual["hidden_dim_1"] and meta_h2 == actual["hidden_dim_2"]:
        result["match"] = True
    else:
        result["match"] = False
        result["warning"] = (
            f"Architecture mismatch between metadata.json and .pt tensor shapes at "
            f"'{artifact_dir}':\n"
            f"  metadata: hidden_dim_1={meta_h1}, hidden_dim_2={meta_h2}\n"
            f"  actual:   hidden_dim_1={actual['hidden_dim_1']}, "
            f"hidden_dim_2={actual['hidden_dim_2']}\n"
            f"Stage 2 will use the ACTUAL tensor shapes for safe loading."
        )
    return result


def build_stage2_model_for_init(
    input_dim: int,
    config: Dict[str, Any],
    init_source_dir: Optional[Path],
    init_name: str,
    model_filename: str = "mlp_pretrained.pt",
    device: Optional[torch.device] = None,
) -> TabularMLP:
    """Build a Stage 2 TabularMLP with the architecture that matches its init.

    This is the **only correct way** to build a model before loading Stage 1
    weights with ``strict=True``.  The architecture is resolved as follows:

    * **Pretrained inits** (A_weak_only / B_gold_only / C_hybrid /
      D_hybrid_unweighted):

      1. Reads ``mlp_architecture`` from the artifact's ``metadata.json``.
      2. Falls back to inferring shapes from the actual ``.pt`` file if
         metadata is absent or stale (emits a warning in that case).
      3. Raises a clear ``ValueError`` if neither source is available.

    * **Random init**: uses the ``mlp_config`` section of
      ``stage2_config.yaml``.  No Stage 1 weights are loaded for random
      init; this is an intentional clean baseline that keeps the feature
      space consistent (by reusing the Stage 1 preprocessor) while
      starting from random parameter values.

    Parameters
    ----------
    input_dim : int
        Number of input features *after* preprocessing (the actual first-layer
        width seen by the MLP, not the raw feature count).
    config : dict
        Resolved Stage 2 experiment config (output of
        :func:`normalize_stage2_config`).
    init_source_dir : Path or None
        Canonical Stage 1 artifact directory.  May be ``None`` only when
        ``init_name == "random"``.
    init_name : str
        One of ``A_weak_only``, ``B_gold_only``, ``C_hybrid``,
        ``D_hybrid_unweighted``, or ``random``.
    model_filename : str
        Filename of the model weights file inside ``init_source_dir``.
    device : torch.device or None
        Target device.  Defaults to CUDA if available, else CPU.

    Returns
    -------
    TabularMLP
        An untrained (or about to be weight-loaded) model on ``device``.

    Raises
    ------
    ValueError
        If ``init_source_dir`` is required but missing, or if neither
        metadata nor the actual state_dict can supply architecture info.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if init_name == RANDOM_INIT_NAME:
        # ── Random init: clean baseline, no Stage 1 weights loaded ──────────
        # Architecture comes from stage2_config.yaml's mlp_config.
        # The preprocessor is still shared with A_weak_only so the input
        # dimensionality is identical across all init conditions.
        mlp_cfg = config["mlp_config"]
        return TabularMLP(
            input_dim=input_dim,
            hidden_dim_1=mlp_cfg["hidden_dim_1"],
            hidden_dim_2=mlp_cfg["hidden_dim_2"],
            dropout=mlp_cfg["dropout"],
        ).to(device)

    # ── Pretrained init: architecture MUST match the stored weights ──────────
    if init_source_dir is None:
        raise ValueError(
            f"init_source_dir must be provided for pretrained init '{init_name}'; "
            f"got None."
        )

    # Attempt 1: read from metadata.json (fast, no torch needed).
    try:
        meta = load_stage1_artifact_metadata(init_source_dir)
        mlp_arch = meta.get("mlp_architecture")
    except FileNotFoundError:
        meta = None
        mlp_arch = None

    if mlp_arch is not None:
        h1 = int(mlp_arch["hidden_dim_1"])
        h2 = int(mlp_arch["hidden_dim_2"])
        dropout = float(mlp_arch.get("dropout", 0.0))
    else:
        # Attempt 2: infer from actual .pt tensor shapes.
        model_path = init_source_dir / model_filename
        if not model_path.exists():
            raise ValueError(
                f"Cannot determine architecture for init '{init_name}': "
                f"metadata.json is missing from '{init_source_dir}' and "
                f"the model file '{model_path}' does not exist."
            )
        try:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            inferred = infer_arch_from_state_dict(state_dict)
            h1 = inferred["hidden_dim_1"]
            h2 = inferred["hidden_dim_2"]
            mlp_cfg_fallback = config.get("mlp_config", {})
            dropout = float(mlp_cfg_fallback.get("dropout", 0.0))
            import warnings
            warnings.warn(
                f"[build_stage2_model_for_init] metadata.json missing at "
                f"'{init_source_dir}'; inferred architecture from .pt tensor "
                f"shapes: hidden_dim_1={h1}, hidden_dim_2={h2}. "
                f"Re-run Stage 1 training to regenerate metadata.",
                UserWarning,
                stacklevel=2,
            )
        except Exception as exc:
            raise ValueError(
                f"Cannot determine architecture for init '{init_name}' at "
                f"'{init_source_dir}': metadata.json is absent and the .pt "
                f"file could not be parsed ({exc}). "
                f"Re-run Stage 1 training to regenerate both artifacts."
            ) from exc

    return TabularMLP(
        input_dim=input_dim,
        hidden_dim_1=h1,
        hidden_dim_2=h2,
        dropout=dropout,
    ).to(device)


def load_stage1_model_weights(
    model: torch.nn.Module,
    model_path: Optional[Path],
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    if model_path is None:
        return model
    if not model_path.exists():
        raise FileNotFoundError(f"Stage 1 model not found: {model_path}")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model


def prepare_stage2_tasks(
    config: Dict[str, Any],
    raw_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    config = normalize_stage2_config(config)
    if raw_data is None:
        raw_data = load_stage2_raw_data(config)

    paths = raw_data["paths"]
    df = raw_data["df"]
    feature_cols = raw_data["feature_cols"]

    task_df = build_department_task_table(
        features_df=df,
        feature_cols=feature_cols,
        contract_id_col=config["data"]["group_col"],
        department_col=config["data"]["department_col"],
        target_col=config["data"]["target_col"],
        observation_year_col=config["data"]["observation_year_col"],
        drop_missing_features=False,
    )

    summary_df = summarize_department_tasks(
        task_df=task_df,
        department_col=config["data"]["department_col"],
        target_col=config["data"]["target_col"],
        contract_id_col=config["data"]["group_col"],
    )

    filtered_task_df, valid_departments, validity_df = filter_valid_departments(
        task_df=task_df,
        department_col=config["data"]["department_col"],
        target_col=config["data"]["target_col"],
        contract_id_col=config["data"]["group_col"],
        n_support_pos=config["support_config"]["n_support_pos"],
        n_support_neg=config["support_config"]["n_support_neg"],
        min_query_contracts=config["task_config"]["min_query_contracts"],
        require_both_query_classes=config["task_config"]["require_both_query_classes"],
    )

    # ── Target-aware meta-train / meta-test split ──────────────────────────
    # The held-out target department is excluded from meta-training and is
    # the *only* department used for meta-test adaptation/evaluation.  This
    # is the same rule applied across all four Stage 2 methods (zero-shot,
    # ANIL, FOMAML, MAML) and is the reason Stage 2 results can be read as a
    # genuine few-shot generalisation test rather than within-distribution
    # performance on departments the model already saw during meta-training.
    target_department = config["task_config"]["target_department"]
    exclude_departments = set(config["task_config"].get("exclude_departments", []))
    meta_train_departments = [
        dept
        for dept in valid_departments
        if dept != target_department and dept not in exclude_departments
    ]
    # Defensive assertion — fail loudly if the held-out department somehow
    # leaks into meta-training (e.g. through future refactors of
    # filter_valid_departments).
    assert target_department not in meta_train_departments, (
        f"Target department '{target_department}' must be excluded from "
        f"meta-training but appears in meta_train_departments."
    )

    # ── Synthetic-augmentation-aware episode generation ────────────────────
    # When synthetic_config specifies an active augmentation method,
    # generate episodes via `sample_episode_with_synthetic_support` so that
    # synthetic support rows are actually added.  Without this branch the
    # augmentation config is saved in resolved_config.yaml but never applied,
    # making gen-smote_nc identical to gen-none.
    syn_cfg = config.get("synthetic_config") or {}
    _aug_method = syn_cfg.get("augmentation_method", "none")
    _syn_prop = float(syn_cfg.get("synthetic_proportion", 0.0))
    _target_per_class = syn_cfg.get("target_per_class", None)
    _aug_seed_offset = int(syn_cfg.get("augmentation_seed_offset", 10007))
    _use_augmentation = (
        _aug_method != "none"
        and (_syn_prop > 0.0 or _target_per_class is not None)
    )

    try:
        if _use_augmentation:
            _target_df = task_df[
                task_df[config["data"]["department_col"]] == target_department
            ].copy()
            _n_repeats = config["meta_config"]["n_repeats"]
            _base_seed = config.get("seed", 42) + _aug_seed_offset
            target_episodes = []
            for _ep_idx in range(_n_repeats):
                _ep = sample_episode_with_synthetic_support(
                    dept_df=_target_df,
                    feature_cols=feature_cols,
                    department_name=target_department,
                    target_col=config["data"]["target_col"],
                    department_col=config["data"]["department_col"],
                    contract_id_col=config["data"]["group_col"],
                    n_support_pos=config["support_config"]["n_support_pos"],
                    n_support_neg=config["support_config"]["n_support_neg"],
                    query_strategy="remainder",
                    augmentation_method=_aug_method,
                    synthetic_proportion=_syn_prop if _target_per_class is None else None,
                    target_per_class=_target_per_class,
                    categorical_cols=syn_cfg.get("categorical_cols", None),
                    k_neighbors=int(syn_cfg.get("k_neighbors", 5)),
                    random_state=_base_seed + _ep_idx,
                )
                _ep["repeat_idx"] = _ep_idx
                _ep["episode_seed"] = _base_seed + _ep_idx
                target_episodes.append(_ep)
        else:
            target_episodes = make_target_meta_test_split(
                task_df=task_df,
                feature_cols=feature_cols,
                target_department=target_department,
                department_col=config["data"]["department_col"],
                target_col=config["data"]["target_col"],
                contract_id_col=config["data"]["group_col"],
                n_support_pos=config["support_config"]["n_support_pos"],
                n_support_neg=config["support_config"]["n_support_neg"],
                n_repeats=config["meta_config"]["n_repeats"],
                require_both_query_classes=config["task_config"].get(
                    "require_both_query_classes", False
                ),
            )
    except ValueError:
        target_episodes = []

    # ── Per-episode support accounting ────────────────────────────────────────
    # Compute real/synthetic row counts from each episode's support_df so the
    # pipeline can write support_accounting.csv without needing episode_results
    # to be embedded in the method's return dict (which it never is).
    try:
        from master_thesis.stage2b_synthetic import compute_support_accounting as _csa
        _target_col = config["data"]["target_col"]
        _group_col = config["data"]["group_col"]
        _episode_accounting: List[Dict[str, Any]] = []
        for _ep_idx, _ep in enumerate(target_episodes):
            _support = _ep.get("support_df") if isinstance(_ep, dict) else getattr(_ep, "support_df", None)
            _query = _ep.get("query_df") if isinstance(_ep, dict) else getattr(_ep, "query_df", None)
            if isinstance(_support, pd.DataFrame) and not _support.empty:
                _acct = _csa(
                    support_df=_support,
                    target_col=_target_col,
                    contract_id_col=_group_col,
                )
            else:
                _acct = {
                    "real_k_pos": 0, "real_k_neg": 0,
                    "synthetic_k_pos": 0, "synthetic_k_neg": 0,
                    "effective_k_pos": 0, "effective_k_neg": 0,
                    "n_real_support_rows": 0, "n_synthetic_support_rows": 0,
                    "n_effective_support_rows": 0,
                    "n_real_support_contracts": 0, "n_synthetic_support_contracts": 0,
                }
            _acct["episode_idx"] = _ep_idx
            _acct["augmentation_method"] = _aug_method
            _acct["synthetic_proportion"] = _syn_prop
            # ── Per-episode gen_skip_reason ─────────────────────────────────
            # Record WHY synthetic rows were not generated when augmentation
            # was requested but produced zero synthetic rows.
            _n_syn = _acct.get("n_synthetic_support_rows", 0) or 0
            _ep_skip_reason: Optional[str] = None
            if _aug_method != "none" and _n_syn == 0:
                _real_pos = _acct.get("real_k_pos", 0) or 0
                _real_neg = _acct.get("real_k_neg", 0) or 0
                if _target_per_class is not None and (
                    _real_pos >= _target_per_class and _real_neg >= _target_per_class
                ):
                    # Both classes already have at least target_per_class rows;
                    # augment_support() correctly computed n_class_synth=0 for both.
                    _ep_skip_reason = "real_support_exceeds_target_effective_k"
                elif _target_per_class is not None and (
                    _real_pos < 2 or _real_neg < 2
                ):
                    # SMOTENC needs ≥2 samples per class to interpolate.
                    _ep_skip_reason = "insufficient_samples_for_smotenc"
                else:
                    _ep_skip_reason = "no_synthetic_rows_generated"
            _acct["gen_skip_reason"] = _ep_skip_reason
            # Query accounting (real rows only by design)
            if isinstance(_query, pd.DataFrame) and not _query.empty:
                _y_q = _query[_target_col] if _target_col in _query.columns else pd.Series(dtype=float)
                _acct["n_query_rows"] = len(_query)
                _acct["n_query_pos"] = int((_y_q == 1).sum())
                _acct["n_query_neg"] = int((_y_q == 0).sum())
                _acct["n_query_synthetic_rows"] = 0  # always real-only
            else:
                _acct["n_query_rows"] = 0
                _acct["n_query_pos"] = 0
                _acct["n_query_neg"] = 0
                _acct["n_query_synthetic_rows"] = 0
            _episode_accounting.append(_acct)
    except Exception:
        _episode_accounting = []

    return {
        "paths": paths,
        "raw_df": df,
        "task_df": task_df,
        "summary_df": summary_df,
        "filtered_task_df": filtered_task_df,
        "valid_departments": valid_departments,
        "validity_df": validity_df,
        "meta_train_departments": meta_train_departments,
        "target_department": target_department,
        "target_episodes": target_episodes,
        "feature_cols": feature_cols,
        "episode_accounting": _episode_accounting,
        "augmentation_method": _aug_method,
        "use_augmentation": _use_augmentation,
    }


def run_finetune_baseline(config: Dict[str, Any], prepared: Dict[str, Any]) -> Dict[str, Any]:
    from master_thesis.metrics import evaluate_on_gold_binary

    paths = prepared["paths"]
    target_episodes = prepared["target_episodes"]
    feature_cols = prepared["feature_cols"]
    target_col = config["data"]["target_col"]

    preprocessor = load_stage1_preprocessor(paths.stage1_preprocessor_path)
    if preprocessor is None:
        raise ValueError("Fine-tuning baseline requires a Stage 1 preprocessor.")

    if len(target_episodes) == 0:
        return {
            "method": "finetune",
            "status": "no_target_episodes_available",
            "target_department": prepared["target_department"],
            "raw_metrics": None,
            "predictions": None,
            "history": None,
        }

    inner_lr = config["meta_config"]["inner_lr"]
    target_inner_steps = config["meta_config"].get("target_inner_steps", config["meta_config"]["inner_steps"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dummy_processed = preprocessor.transform(target_episodes[0]["support_df"][feature_cols].iloc[:2])
    if hasattr(dummy_processed, "toarray"):
        dummy_processed = dummy_processed.toarray()
    init_name = config["stage1_init"]["init_name"]
    base_model = build_stage2_model_for_init(
        input_dim=dummy_processed.shape[1],
        config=config,
        init_source_dir=paths.init_source_dir,
        init_name=init_name,
        device=device,
    )
    base_model = load_stage1_model_weights(base_model, paths.stage1_model_path, device)

    all_metrics, all_predictions = [], []
    criterion = torch.nn.BCEWithLogitsLoss()

    for ep_idx, episode in enumerate(target_episodes):
        X_supp = preprocessor.transform(episode["support_df"][feature_cols])
        X_query = preprocessor.transform(episode["query_df"][feature_cols])
        if hasattr(X_supp, "toarray"):
            X_supp = X_supp.toarray()
        if hasattr(X_query, "toarray"):
            X_query = X_query.toarray()

        y_supp = torch.tensor(episode["support_df"][target_col].to_numpy(), dtype=torch.float32, device=device).view(-1, 1)
        y_query_np = episode["query_df"][target_col].to_numpy()

        X_supp_t = torch.tensor(X_supp, dtype=torch.float32, device=device)
        X_query_t = torch.tensor(X_query, dtype=torch.float32, device=device)

        adapted_model = copy.deepcopy(base_model)
        adapted_model.train()
        optimizer = torch.optim.Adam(adapted_model.parameters(), lr=inner_lr)

        for _ in range(target_inner_steps):
            optimizer.zero_grad()
            loss = criterion(adapted_model(X_supp_t), y_supp)
            loss.backward()
            optimizer.step()

        adapted_model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(adapted_model(X_query_t)).cpu().numpy().ravel()

        metrics = evaluate_on_gold_binary(y_query_np, probs, model_name="FINETUNE")
        metrics["episode_idx"] = ep_idx
        all_metrics.append(metrics)
        all_predictions.append(pd.DataFrame({"episode_idx": ep_idx, "y_true": y_query_np, "y_prob": probs}))

    return {
        "method": "finetune",
        "status": "success",
        "target_department": prepared["target_department"],
        "n_target_episodes": len(target_episodes),
        "raw_metrics": pd.concat(all_metrics, ignore_index=True),
        "predictions": pd.concat(all_predictions, ignore_index=True),
        "history": None,
    }


def _run_meta_method(
    *,
    method_name: str,
    meta_train_fn,
    eval_fn,
    weight_filename: str,
    config: Dict[str, Any],
    prepared: Dict[str, Any],
) -> Dict[str, Any]:
    feature_cols = prepared["feature_cols"]
    task_df = prepared["filtered_task_df"]
    meta_train_departments = prepared["meta_train_departments"]
    target_episodes = prepared["target_episodes"]
    paths = prepared["paths"]

    target_col = config["data"]["target_col"]
    department_col = config["data"]["department_col"]
    contract_id_col = config["data"]["group_col"]
    meta_cfg = config["meta_config"]

    preprocessor = load_stage1_preprocessor(paths.stage1_preprocessor_path)
    if preprocessor is None:
        raise ValueError(f"{method_name.upper()} requires a Stage 1 preprocessor.")

    dummy_processed = preprocessor.transform(task_df[feature_cols].iloc[:2])
    if hasattr(dummy_processed, "toarray"):
        dummy_processed = dummy_processed.toarray()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_name = config["stage1_init"]["init_name"]
    model = build_stage2_model_for_init(
        input_dim=dummy_processed.shape[1],
        config=config,
        init_source_dir=paths.init_source_dir,
        init_name=init_name,
        device=device,
    )
    model = load_stage1_model_weights(model, paths.stage1_model_path, device)

    meta_train_result = meta_train_fn(
        model=model,
        preprocessor=preprocessor,
        task_df=task_df,
        feature_cols=feature_cols,
        meta_train_departments=meta_train_departments,
        target_col=target_col,
        department_col=department_col,
        contract_id_col=contract_id_col,
        meta_batch_size=meta_cfg["meta_batch_size"],
        meta_iterations=meta_cfg["meta_iterations"],
        inner_lr=meta_cfg["inner_lr"],
        outer_lr=meta_cfg["outer_lr"],
        inner_steps=meta_cfg["inner_steps"],
        n_support_pos=config["support_config"].get("n_support_pos", 1),
        n_support_neg=config["support_config"].get("n_support_neg", 1),
        device=device,
        random_state=config["seed"],
        gradient_clip_norm=meta_cfg.get("gradient_clip_norm", 1.0),
        inner_grad_clip=meta_cfg.get("inner_grad_clip", 1.0),
    )

    trained_model = meta_train_result["model"]
    save_path = paths.output_dir / weight_filename
    torch.save(trained_model.state_dict(), save_path)

    eval_result = eval_fn(
        model=trained_model,
        preprocessor=preprocessor,
        target_episodes=target_episodes,
        feature_cols=feature_cols,
        target_col=target_col,
        inner_lr=meta_cfg["inner_lr"],
        target_inner_steps=meta_cfg.get("target_inner_steps", meta_cfg["inner_steps"]),
        device=device,
        inner_grad_clip=meta_cfg.get("inner_grad_clip", 1.0),
    )

    return {
        "method": method_name,
        "status": meta_train_result.get("status", "success"),
        "target_department": prepared["target_department"],
        "weight_path": str(save_path),
        "raw_metrics": eval_result.get("raw_metrics"),
        "predictions": eval_result.get("predictions"),
        "history": meta_train_result.get("history"),
        "diagnostics": meta_train_result.get("diagnostics"),
    }


def run_anil_method(config: Dict[str, Any], prepared: Dict[str, Any]) -> Dict[str, Any]:
    return _run_meta_method(
        method_name="anil",
        meta_train_fn=meta_train_anil,
        eval_fn=evaluate_anil_on_target_episodes,
        weight_filename="anil_initialized.pt",
        config=config,
        prepared=prepared,
    )


def run_fomaml_method(config: Dict[str, Any], prepared: Dict[str, Any]) -> Dict[str, Any]:
    return _run_meta_method(
        method_name="fomaml",
        meta_train_fn=meta_train_fomaml,
        eval_fn=evaluate_fomaml_on_target_episodes,
        weight_filename="fomaml_initialized.pt",
        config=config,
        prepared=prepared,
    )


def run_maml_method(config: Dict[str, Any], prepared: Dict[str, Any]) -> Dict[str, Any]:
    return _run_meta_method(
        method_name="maml",
        meta_train_fn=meta_train_maml,
        eval_fn=evaluate_maml_on_target_episodes,
        weight_filename="maml_initialized.pt",
        config=config,
        prepared=prepared,
    )


def run_zero_shot_method(config: Dict[str, Any], prepared: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate the Stage 1 model directly on target episodes (no adaptation).

    This is the pure no-adapt floor: the Stage 1 MLP is loaded with its
    pretrained weights and queried on each target episode's query set. It is
    the correct reference point for every adaptive method (finetune, ANIL,
    FOMAML, MAML) because it isolates the contribution of Stage 2 adaptation
    from the underlying pretrained representation.
    """
    paths = prepared["paths"]
    target_episodes = prepared["target_episodes"]
    feature_cols = prepared["feature_cols"]
    target_col = config["data"]["target_col"]

    preprocessor = load_stage1_preprocessor(paths.stage1_preprocessor_path)
    if preprocessor is None:
        raise ValueError("Zero-shot evaluation requires a Stage 1 preprocessor.")

    if len(target_episodes) == 0:
        return {
            "method": "zero_shot",
            "status": "no_target_episodes_available",
            "target_department": prepared["target_department"],
            "raw_metrics": None,
            "predictions": None,
            "history": None,
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dummy_processed = preprocessor.transform(target_episodes[0]["support_df"][feature_cols].iloc[:2])
    if hasattr(dummy_processed, "toarray"):
        dummy_processed = dummy_processed.toarray()
    init_name = config["stage1_init"]["init_name"]
    base_model = build_stage2_model_for_init(
        input_dim=dummy_processed.shape[1],
        config=config,
        init_source_dir=paths.init_source_dir,
        init_name=init_name,
        device=device,
    )
    base_model = load_stage1_model_weights(base_model, paths.stage1_model_path, device)

    eval_result = evaluate_zero_shot_on_target_episodes(
        model=base_model,
        preprocessor=preprocessor,
        target_episodes=target_episodes,
        feature_cols=feature_cols,
        target_col=target_col,
        device=device,
        method_name="zero_shot",
        contract_id_col=config["data"].get("group_col", "contract_id"),
        department_col=config["data"].get("department_col", "department"),
    )

    return {
        "method": "zero_shot",
        "status": eval_result.get("status", "success"),
        "target_department": prepared["target_department"],
        "n_target_episodes": eval_result.get("n_target_episodes", 0),
        "raw_metrics": eval_result.get("raw_metrics"),
        "predictions": eval_result.get("predictions"),
        "prediction_stats": eval_result.get("prediction_stats"),
        "history": None,
    }


def run_stage2_method(config: Dict[str, Any], prepared: Dict[str, Any]) -> Dict[str, Any]:
    method = config["method"].lower()
    if method == "zero_shot":
        return run_zero_shot_method(config, prepared)
    if method == "finetune":
        return run_finetune_baseline(config, prepared)
    if method == "anil":
        return run_anil_method(config, prepared)
    if method == "fomaml":
        return run_fomaml_method(config, prepared)
    if method == "maml":
        return run_maml_method(config, prepared)
    raise ValueError(f"Unsupported Stage 2 method: {method}")


def summarize_stage2_result(
    result: Dict[str, Any],
    config: Dict[str, Any],
    experiment_id: Optional[str] = None,
) -> Dict[str, Any]:
    syn_cfg = config.get("synthetic_config") or {}
    acct = result.get("accounting_summary") or {}
    summary = {
        "experiment_id": experiment_id,
        "method": config.get("method"),
        "init_name": config.get("stage1_init", {}).get("init_name"),
        "target_department": config.get("task_config", {}).get("target_department"),
        "n_support_pos": config.get("support_config", {}).get("n_support_pos"),
        "n_support_neg": config.get("support_config", {}).get("n_support_neg"),
        "inner_steps": config.get("meta_config", {}).get("inner_steps"),
        "inner_lr": config.get("meta_config", {}).get("inner_lr"),
        "meta_batch_size": config.get("meta_config", {}).get("meta_batch_size"),
        "gradient_clip_norm": config.get("meta_config", {}).get("gradient_clip_norm"),
        "inner_grad_clip": config.get("meta_config", {}).get("inner_grad_clip"),
        "synthetic_proportion": syn_cfg.get("synthetic_proportion", 0.0),
        "augmentation_method": syn_cfg.get("augmentation_method", "none"),
        "target_per_class": syn_cfg.get("target_per_class"),
        "k_neighbors": syn_cfg.get(
            "k_neighbors", DEFAULT_SYNTHETIC_CONFIG["k_neighbors"]
        ),
        "status": result.get("status"),
        # ── Support accounting fields ────────────────────────────────────────
        # augmentation_method_requested: what the config asked for
        "augmentation_method_requested": syn_cfg.get("augmentation_method", "none"),
        # augmentation_method_actual: what was applied (may differ if fallback)
        "augmentation_method_actual": acct.get(
            "augmentation_method_actual", syn_cfg.get("augmentation_method", "none")
        ),
        # target_effective_k mirrors target_per_class (the resolved per-class cap)
        "target_effective_k": syn_cfg.get("target_per_class"),
        # Mean real/synthetic support row counts across episodes
        "n_real_support_rows": acct.get("n_real_support_rows_mean"),
        "n_synthetic_support_rows": acct.get("n_synthetic_support_rows_mean"),
        "n_effective_support_rows": acct.get("n_effective_support_rows_mean"),
        "n_real_support_pos": acct.get("real_k_pos_mean"),
        "n_real_support_neg": acct.get("real_k_neg_mean"),
        "n_synthetic_support_pos": acct.get("synthetic_k_pos_mean"),
        "n_synthetic_support_neg": acct.get("synthetic_k_neg_mean"),
        # Query accounting (real-only by design)
        "n_query_rows": acct.get("n_query_rows_mean"),
        "n_query_pos": acct.get("n_query_pos_mean"),
        "n_query_neg": acct.get("n_query_neg_mean"),
        "n_query_synthetic_rows": 0,  # invariant: query is always real-only
        # Augmentation health flags
        "gen_fallback_used": acct.get("gen_fallback_used"),
        "gen_skip_reason": acct.get("gen_skip_reason"),
    }

    raw_metrics = result.get("raw_metrics")
    if isinstance(raw_metrics, pd.DataFrame) and not raw_metrics.empty:
        numeric_cols = [
            col for col in raw_metrics.columns
            if pd.api.types.is_numeric_dtype(raw_metrics[col]) and col != "episode_idx"
        ]
        for col in numeric_cols:
            summary[f"{col}_mean"] = float(raw_metrics[col].mean())
            summary[f"{col}_std"] = float(raw_metrics[col].std(ddof=0))
        summary["n_metric_rows"] = int(len(raw_metrics))
    else:
        summary["n_metric_rows"] = 0

    predictions = result.get("predictions")
    summary["n_predictions"] = int(len(predictions)) if isinstance(predictions, pd.DataFrame) else 0

    # Post-hoc collapse diagnostics (method-agnostic; label-free).
    pred_stats = result.get("prediction_stats")
    if not isinstance(pred_stats, pd.DataFrame) and isinstance(predictions, pd.DataFrame) and not predictions.empty:
        try:
            from master_thesis.meta_common import prediction_stats_frame_from_predictions
            pred_stats = prediction_stats_frame_from_predictions(predictions)
        except Exception:
            pred_stats = None

    if isinstance(pred_stats, pd.DataFrame) and not pred_stats.empty:
        summary["n_episodes_diagnosed"] = int(len(pred_stats))
        if "collapsed" in pred_stats.columns:
            summary["collapse_rate"] = float(pred_stats["collapsed"].mean())
        for col in ("pred_mean", "pred_std", "frac_near_zero", "frac_near_one"):
            if col in pred_stats.columns:
                summary[f"{col}_mean"] = float(pred_stats[col].mean())

    # ── Episode-level class diagnostics ──────────────────────────────────────
    # Compute per-episode query and support class counts from predictions and
    # the prepared data (when available).
    if isinstance(predictions, pd.DataFrame) and not predictions.empty and "y_true" in predictions.columns:
        ep_groups = predictions.groupby("episode_idx")["y_true"]
        query_true_means = ep_groups.mean()
        query_pos_counts = ep_groups.sum()
        query_neg_counts = ep_groups.apply(lambda s: (s == 0).sum())

        summary["query_true_mean_mean"] = float(query_true_means.mean())
        summary["query_pos_count_mean"] = float(query_pos_counts.mean())
        summary["query_neg_count_mean"] = float(query_neg_counts.mean())

        # Count episodes where query has only one class.
        n_one_class_query = int(((query_pos_counts == 0) | (query_neg_counts == 0)).sum())
        summary["n_one_class_query_episodes"] = n_one_class_query

    # Support class counts from episode_accounting (computed during prepare).
    if isinstance(result.get("episode_accounting"), list) and len(result["episode_accounting"]) > 0:
        acct_list = result["episode_accounting"]
        support_pos = [a.get("real_k_pos", 0) + a.get("synthetic_k_pos", 0) for a in acct_list]
        support_neg = [a.get("real_k_neg", 0) + a.get("synthetic_k_neg", 0) for a in acct_list]
        support_total = [p + n for p, n in zip(support_pos, support_neg)]
        summary["support_pos_count_mean"] = float(np.mean(support_pos)) if support_pos else float("nan")
        summary["support_neg_count_mean"] = float(np.mean(support_neg)) if support_neg else float("nan")
        if support_total:
            summary["support_true_mean_mean"] = float(
                np.mean([p / max(t, 1) for p, t in zip(support_pos, support_total)])
            )

    # ── Stability warnings ───────────────────────────────────────────────────
    warnings_list = []
    if summary.get("n_one_class_query_episodes", 0) > 0:
        warnings_list.append(
            f"WARNING: {summary['n_one_class_query_episodes']} query episode(s) "
            f"have only one class (AUROC undefined)"
        )
    if summary.get("query_true_mean_mean", 0.0) > 0.85:
        warnings_list.append(
            f"WARNING: query_true_mean_mean={summary['query_true_mean_mean']:.3f} "
            f"> 0.85 — query is severely positive-heavy"
        )
    pred_mean_val = summary.get("gold_pred_mean_mean") or summary.get("pred_mean_mean", 0.0)
    pred_std_val = summary.get("gold_pred_std_mean") or summary.get("pred_std_mean", 1.0)
    if pred_mean_val is not None and pred_std_val is not None:
        if pred_mean_val > 0.95 and pred_std_val < 0.01:
            warnings_list.append(
                f"WARNING: pred_mean={pred_mean_val:.4f}, pred_std={pred_std_val:.4f} "
                f"— predictions are collapsed"
            )
    if warnings_list:
        summary["stability_warnings"] = "; ".join(warnings_list)

    return summary


def save_stage2_tables(prepared: Dict[str, Any], output_dir: Path) -> None:
    prepared["summary_df"].to_csv(output_dir / "department_summary.csv", index=False)
    prepared["validity_df"].to_csv(output_dir / "department_validity.csv", index=False)


def _make_serializable(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.nn.Module):
        return f"<torch.nn.Module: {obj.__class__.__name__}>"
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def save_stage2_result(
    result: Dict[str, Any],
    config: Dict[str, Any],
    output_dir: Path,
    experiment_id: Optional[str] = None,
) -> None:
    with open(output_dir / "resolved_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    if isinstance(result.get("raw_metrics"), pd.DataFrame):
        result["raw_metrics"].to_csv(output_dir / "metrics.csv", index=False)
    if isinstance(result.get("predictions"), pd.DataFrame):
        result["predictions"].to_csv(output_dir / "predictions.csv", index=False)
    if isinstance(result.get("history"), pd.DataFrame):
        result["history"].to_csv(output_dir / "history.csv", index=False)

    # Post-hoc prediction-stats / collapse diagnostic (method-agnostic).
    # Computed from `predictions` if not already supplied by the method.
    predictions_df = result.get("predictions")
    pred_stats = result.get("prediction_stats")
    if not isinstance(pred_stats, pd.DataFrame):
        if isinstance(predictions_df, pd.DataFrame) and not predictions_df.empty:
            try:
                from master_thesis.meta_common import prediction_stats_frame_from_predictions
                pred_stats = prediction_stats_frame_from_predictions(predictions_df)
            except Exception:
                pred_stats = None
    if isinstance(pred_stats, pd.DataFrame) and not pred_stats.empty:
        pred_stats.to_csv(output_dir / "prediction_stats.csv", index=False)

    # ── Support accounting CSV ─────────────────────────────────────────────
    # episode_accounting is injected into result by run_stage2_pipeline()
    # from the prepared dict (computed in prepare_stage2_tasks).  This
    # replaces the old dead-code path that checked result["episode_results"]
    # — a key that _run_meta_method and run_finetune_baseline never populate.
    _episode_accounting = result.get("episode_accounting", [])
    if _episode_accounting:
        _acct_df = pd.DataFrame(_episode_accounting)
        _acct_df.to_csv(output_dir / "support_accounting.csv", index=False)

    # ── stage2b_config.json ────────────────────────────────────────────────
    # Write a machine-readable audit file for every run that touched synthetic
    # augmentation (including gen-none runs in a synth sweep, so the folder
    # always has this file for easy programmatic checks).
    syn_cfg = config.get("synthetic_config") or {}
    _aug_method_cfg = syn_cfg.get("augmentation_method", "none")
    _is_synth_sweep = (
        _aug_method_cfg != "none"
        or float(syn_cfg.get("synthetic_proportion", 0.0)) > 0.0
        or syn_cfg.get("target_per_class") is not None
    )
    if _is_synth_sweep:
        _accounting_summary = result.get("accounting_summary") or {}
        _stage2b_cfg: Dict[str, Any] = {
            "augmentation_method_requested": _aug_method_cfg,
            "augmentation_method_actual": _accounting_summary.get(
                "augmentation_method_actual", _aug_method_cfg
            ),
            "synthetic_proportion": syn_cfg.get("synthetic_proportion", 0.0),
            "target_per_class": syn_cfg.get("target_per_class"),
            "k_neighbors": syn_cfg.get("k_neighbors"),
            "categorical_cols": syn_cfg.get("categorical_cols"),
            "augmentation_seed_offset": syn_cfg.get("augmentation_seed_offset"),
            "gen_fallback_used": _accounting_summary.get("gen_fallback_used"),
            "gen_skip_reason": _accounting_summary.get("gen_skip_reason"),
            "n_real_support_rows_mean": _accounting_summary.get("n_real_support_rows_mean"),
            "n_synthetic_support_rows_mean": _accounting_summary.get("n_synthetic_support_rows_mean"),
        }
        with open(output_dir / "stage2b_config.json", "w", encoding="utf-8") as _f:
            json.dump(_make_serializable(_stage2b_cfg), _f, indent=2)

    summary = summarize_stage2_result(result, config, experiment_id)
    with open(output_dir / "stage2_result_summary.json", "w", encoding="utf-8") as f:
        json.dump(_make_serializable(summary), f, indent=2)
    with open(output_dir / "stage2_result.json", "w", encoding="utf-8") as f:
        json.dump(_make_serializable(result), f, indent=2)


def backfill_legacy_stage2_results(
    models_stage2_dir: Optional[Path] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Scan legacy Stage 2 result directories and backfill them into the
    grid-based experiment registry.

    Legacy format: ``models/stage_2/{init_name}/{method}/``
    Current format: ``models/stage_2/experiments/{experiment_id}/``

    The function is idempotent: experiments already present in the
    registry (matched by experiment_id) are skipped.

    Parameters
    ----------
    models_stage2_dir
        Root of the Stage 2 models directory.  Defaults to MODELS_STAGE2.
    verbose
        Print progress messages.

    Returns
    -------
    pd.DataFrame
        Updated master experiment summary (all experiments, new + existing).
    """
    if models_stage2_dir is None:
        models_stage2_dir = MODELS_STAGE2

    experiment_root = models_stage2_dir / "experiments"
    experiment_root.mkdir(parents=True, exist_ok=True)
    summary_path = experiment_root / "experiment_summary.csv"

    # Load current registry if it exists.
    if summary_path.exists():
        existing_summary = pd.read_csv(summary_path)
        existing_ids: set = set(existing_summary["experiment_id"].dropna().tolist())
    else:
        existing_summary = pd.DataFrame()
        existing_ids = set()

    new_rows: List[Dict[str, Any]] = []

    # Scan directories that look like legacy runs: models/stage_2/{init_name}/{method}/
    # These are NOT inside the "experiments" subfolder.
    for init_dir in models_stage2_dir.iterdir():
        if not init_dir.is_dir() or init_dir.name == "experiments":
            continue
        init_name = init_dir.name
        for method_dir in init_dir.iterdir():
            if not method_dir.is_dir():
                continue

            # Must contain either metrics.csv or stage2_config_snapshot.json to be a valid legacy run.
            metrics_path = method_dir / "metrics.csv"
            snapshot_path = method_dir / "stage2_config_snapshot.json"
            if not metrics_path.exists() and not snapshot_path.exists():
                continue

            # Resolve config from snapshot if available, otherwise use defaults.
            if snapshot_path.exists():
                try:
                    with open(snapshot_path, "r", encoding="utf-8") as f:
                        snap_cfg = json.load(f)
                except Exception:
                    snap_cfg = {}
            else:
                snap_cfg = {}

            method = snap_cfg.get("method") or method_dir.name
            support_cfg = snap_cfg.get("support_config", {})
            n_support_pos = int(support_cfg.get("n_support_pos", 2))
            n_support_neg = int(support_cfg.get("n_support_neg", 2))
            inner_steps = int(
                snap_cfg.get("meta_config", {}).get("inner_steps", 3)
            )
            target_department = (
                snap_cfg.get("task_config", {}).get("target_department", "Logistics")
            )
            resolved_init = snap_cfg.get("stage1_init", {}).get("init_name") or init_name

            exp_id = generate_experiment_id(
                init_name=resolved_init,
                method=method,
                n_support_pos=n_support_pos,
                n_support_neg=n_support_neg,
                inner_steps=inner_steps,
                target_department=target_department,
            )

            if exp_id in existing_ids:
                if verbose:
                    print(f"Already registered, skipping: {exp_id}")
                continue

            # Create destination directory.
            dest_dir = experiment_root / exp_id
            dest_dir.mkdir(parents=True, exist_ok=True)

            # Copy artifacts (do not overwrite if already present).
            artifacts_to_copy = [
                "metrics.csv",
                "predictions.csv",
                "history.csv",
                "department_summary.csv",
                "department_validity.csv",
            ]
            for fname in artifacts_to_copy:
                src = method_dir / fname
                dst = dest_dir / fname
                if src.exists() and not dst.exists():
                    import shutil
                    shutil.copy2(src, dst)

            # Copy weight file (any .pt file).
            for pt_file in method_dir.glob("*.pt"):
                dst_pt = dest_dir / pt_file.name
                if not dst_pt.exists():
                    import shutil
                    shutil.copy2(pt_file, dst_pt)

            # Write resolved_config.yaml from snapshot.
            resolved_config_path = dest_dir / "resolved_config.yaml"
            if not resolved_config_path.exists() and snap_cfg:
                with open(resolved_config_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(snap_cfg, f, sort_keys=False)

            # Reconstruct full summary from metrics.csv.
            summary_row: Dict[str, Any] = {
                "experiment_id": exp_id,
                "method": method,
                "init_name": resolved_init,
                "target_department": target_department,
                "n_support_pos": n_support_pos,
                "n_support_neg": n_support_neg,
                "inner_steps": inner_steps,
                "status": "success",
                "was_skipped": False,
                "backfilled": True,
            }

            if metrics_path.exists():
                try:
                    metrics_df = pd.read_csv(metrics_path)
                    numeric_cols = [
                        c for c in metrics_df.columns
                        if pd.api.types.is_numeric_dtype(metrics_df[c])
                        and c not in ("episode_idx", "repeat_idx")
                    ]
                    for col in numeric_cols:
                        summary_row[f"{col}_mean"] = float(metrics_df[col].mean())
                        summary_row[f"{col}_std"] = float(metrics_df[col].std(ddof=0))
                    summary_row["n_metric_rows"] = int(len(metrics_df))
                except Exception as exc:
                    summary_row["n_metric_rows"] = 0
                    summary_row["metrics_read_error"] = str(exc)
            else:
                summary_row["n_metric_rows"] = 0

            pred_path = method_dir / "predictions.csv"
            if pred_path.exists():
                try:
                    summary_row["n_predictions"] = int(len(pd.read_csv(pred_path)))
                except Exception:
                    summary_row["n_predictions"] = 0
            else:
                summary_row["n_predictions"] = 0

            # Write a canonical stage2_result_summary.json to the new location.
            with open(dest_dir / "stage2_result_summary.json", "w", encoding="utf-8") as f:
                json.dump({k: v for k, v in summary_row.items()}, f, indent=2)

            new_rows.append(summary_row)

            if verbose:
                print(f"Backfilled: {exp_id}")

    if not new_rows:
        if verbose:
            print("No new legacy experiments found to backfill.")
        return existing_summary

    new_df = pd.DataFrame(new_rows)
    if existing_summary.empty:
        updated_summary = new_df
    else:
        updated_summary = pd.concat([existing_summary, new_df], ignore_index=True, sort=False)

    sort_cols = [
        c for c in ["method", "init_name", "n_support_pos", "n_support_neg", "inner_steps", "target_department"]
        if c in updated_summary.columns
    ]
    if sort_cols:
        updated_summary = updated_summary.sort_values(sort_cols).reset_index(drop=True)

    updated_summary.to_csv(summary_path, index=False)

    if verbose:
        print(f"Added {len(new_rows)} legacy experiment(s) to registry.")
        print(f"Saved updated experiment_summary.csv to {summary_path}")

    return updated_summary


def stage1_artifact_inventory(
    stage1_root: Optional[Path] = None,
    model_filename: str = "mlp_pretrained.pt",
    preprocessor_filename: str = "mlp_pretrained_preprocessor.joblib",
) -> Dict[str, Any]:
    """High-level summary of which Stage 1 artifacts are present on disk.

    Wraps :func:`discover_stage1_artifacts` and produces a small report bundle
    suitable for in-notebook display.  Useful as a one-shot diagnostic at the
    top of notebook 17 / 17.1.

    Returns a dict with keys:
      * ``df`` — full per-cell discovery table
      * ``runnable`` — subset filtered to ``is_runnable == True``
      * ``missing`` — subset that is *expected* but not present
      * ``targets_with_full_set`` — sorted list of departments that have all
        three target-aware inits available
    """
    df = discover_stage1_artifacts(
        stage1_root=stage1_root,
        model_filename=model_filename,
        preprocessor_filename=preprocessor_filename,
    )
    if df.empty:
        return {"df": df, "runnable": df, "missing": df, "targets_with_full_set": []}

    runnable = df[df["is_runnable"]].copy()
    missing = df[~df["is_runnable"]].copy()

    targets_with_full_set: List[str] = []
    if "target_department" in runnable.columns:
        target_aware = runnable[runnable["scope"] == "target_aware"]
        for dept, sub in target_aware.groupby("target_department"):
            present_inits = set(sub["init_name"].tolist())
            if set(TARGET_AWARE_INIT_NAMES).issubset(present_inits):
                targets_with_full_set.append(str(dept))
    targets_with_full_set.sort()

    return {
        "df": df,
        "runnable": runnable,
        "missing": missing,
        "targets_with_full_set": targets_with_full_set,
    }


def _aggregate_episode_accounting(episode_accounting: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-episode accounting stats into means for the summary row.

    Numeric columns are averaged across episodes.  The ``gen_fallback_used``
    flag is True when every episode produced zero synthetic rows despite an
    active augmentation method (i.e. SMOTENC fell back to real-only).
    """
    if not episode_accounting:
        return {}
    numeric_keys = [
        "real_k_pos", "real_k_neg",
        "synthetic_k_pos", "synthetic_k_neg",
        "effective_k_pos", "effective_k_neg",
        "n_real_support_rows", "n_synthetic_support_rows", "n_effective_support_rows",
        "n_real_support_contracts", "n_synthetic_support_contracts",
        "n_query_rows", "n_query_pos", "n_query_neg", "n_query_synthetic_rows",
    ]
    agg: Dict[str, Any] = {}
    for k in numeric_keys:
        vals = [ep[k] for ep in episode_accounting if ep.get(k) is not None]
        if vals:
            agg[f"{k}_mean"] = float(sum(vals) / len(vals))
    # Detect fallback: augmentation was requested but no synthetic rows generated
    aug_methods = [ep.get("augmentation_method", "none") for ep in episode_accounting]
    agg["augmentation_method_actual"] = aug_methods[0] if aug_methods else "none"
    syn_counts = [ep.get("n_synthetic_support_rows", 0) or 0 for ep in episode_accounting]
    requested_aug = any(m not in (None, "none") for m in aug_methods)
    agg["gen_fallback_used"] = bool(requested_aug and all(c == 0 for c in syn_counts))

    # Aggregate gen_skip_reason from per-episode reasons.
    # Priority: use the per-episode reasons when all episodes share the same
    # reason (the common case).  Fall back to a generic label only when reasons
    # are mixed or absent.
    if agg["gen_fallback_used"]:
        ep_reasons = [ep.get("gen_skip_reason") for ep in episode_accounting]
        non_null_reasons = [r for r in ep_reasons if r]
        if non_null_reasons:
            unique_reasons = set(non_null_reasons)
            if len(unique_reasons) == 1:
                # All episodes share the same skip reason — use it directly.
                agg["gen_skip_reason"] = unique_reasons.pop()
            else:
                # Mixed reasons across episodes — record all unique ones.
                agg["gen_skip_reason"] = "mixed:" + "|".join(sorted(unique_reasons))
        else:
            # Per-episode reasons were not recorded (pre-fix data or empty).
            agg["gen_skip_reason"] = "smotenc_fallback_all_episodes"
    else:
        agg["gen_skip_reason"] = None
    return agg


def run_stage2_pipeline(
    config: Dict[str, Any],
    experiment_id: Optional[str] = None,
    raw_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    config = normalize_stage2_config(config)
    set_seed(config.get("seed", SEED))

    if raw_data is None:
        raw_data = load_stage2_raw_data(config)

    prepared = prepare_stage2_tasks(config=config, raw_data=raw_data)
    if experiment_id:
        prepared["paths"] = resolve_stage2_paths(config, experiment_id=experiment_id)

    output_dir = prepared["paths"].output_dir
    save_stage2_tables(prepared, output_dir)
    result = run_stage2_method(config, prepared)

    # Inject per-episode accounting computed during prepare_stage2_tasks so
    # save_stage2_result can write support_accounting.csv without relying on
    # episode_results being in the method's return dict (which it never is).
    episode_accounting = prepared.get("episode_accounting", [])
    result["episode_accounting"] = episode_accounting
    result["accounting_summary"] = _aggregate_episode_accounting(episode_accounting)

    save_stage2_result(result, config, output_dir, experiment_id=experiment_id)

    return {
        "prepared": prepared,
        "result": result,
        "summary": summarize_stage2_result(result, config, experiment_id=experiment_id),
    }
