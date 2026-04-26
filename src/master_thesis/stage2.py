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
import pandas as pd
import torch
import yaml

from master_thesis.config import DATA_PROCESSED, MODELS_STAGE1, MODELS_STAGE2, SEED
from master_thesis.episode_sampler import (
    build_department_task_table,
    filter_valid_departments,
    make_logistics_meta_test_split,
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
            row = {
                "experiment_id": exp_id,
                "init_name": cfg["stage1_init"]["init_name"],
                "target_department": cfg["task_config"]["target_department"],
                "method": cfg["method"],
                "init_scope": paths.init_scope,
                "init_source_dir": str(paths.init_source_dir) if paths.init_source_dir else None,
                "model_path": str(paths.stage1_model_path) if paths.stage1_model_path else None,
                "preprocessor_path": str(paths.stage1_preprocessor_path) if paths.stage1_preprocessor_path else None,
                "model_present": (
                    paths.stage1_model_path is not None
                    and paths.stage1_model_path.exists()
                ),
                "preprocessor_present": (
                    paths.stage1_preprocessor_path is not None
                    and paths.stage1_preprocessor_path.exists()
                ),
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
                "resolution_error": str(exc),
            }

        # Random init is allowed to have model_present == False; it just needs
        # a valid preprocessor to define the feature space.
        if row["init_name"] == RANDOM_INIT_NAME:
            row["is_runnable"] = bool(row["preprocessor_present"]) and (row["resolution_error"] is None)
        else:
            row["is_runnable"] = bool(row["model_present"] and row["preprocessor_present"])
        rows.append(row)
        if row["is_runnable"]:
            runnable.append((exp_id, cfg))

    report_df = pd.DataFrame(rows)
    invalid = report_df[~report_df["is_runnable"]] if not report_df.empty else report_df
    if verbose and not invalid.empty:
        print(f"[Stage 2 pre-flight] {len(invalid)} of {len(report_df)} cell(s) are NOT runnable:")
        cols = ["experiment_id", "init_name", "target_department", "method",
                "model_present", "preprocessor_present", "resolution_error"]
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
) -> str:
    """Build a deterministic experiment ID.

    The synthetic-augmentation suffixes (``__synprop-…``, ``__gen-…``) are
    appended only when explicitly provided.  This keeps legacy experiment
    directories (which were generated before synthetic augmentation existed)
    byte-identical, while letting new experiments carve out their own
    directories per ``(syn_prop, gen_method)`` combination.
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
) -> Dict[str, Any]:
    """Build the per-experiment ``synthetic_config`` block.

    Order of precedence (low → high):
      1. ``DEFAULT_SYNTHETIC_CONFIG`` module constant
      2. ``base_config["synthetic_config"]`` if present
      3. ``experiment_grid["synthetic_config"]`` if present (preset-level)
      4. The two per-cell axes (synthetic_proportion, augmentation_method)
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
    return block


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

    preset_meta_overrides = {}
    for key in ["meta_iterations", "n_repeats", "inner_lr", "outer_lr", "meta_batch_size", "target_inner_steps"]:
        if key in experiment_grid:
            preset_meta_overrides[key] = experiment_grid[key]

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
    ) in product(
        methods,
        support_grid,
        inner_steps_grid,
        init_names,
        target_departments,
        inner_lr_grid,
        synprop_grid,
        gen_method_grid,
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

        # Attach the resolved synthetic_config block.
        resolved["synthetic_config"] = _build_synthetic_block(
            base_config=base_config,
            experiment_grid=experiment_grid,
            synthetic_proportion=canonical_synprop,
            augmentation_method=canonical_method,
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

    try:
        target_episodes = make_logistics_meta_test_split(
            task_df=task_df,
            feature_cols=feature_cols,
            target_department=target_department,
            department_col=config["data"]["department_col"],
            target_col=config["data"]["target_col"],
            contract_id_col=config["data"]["group_col"],
            n_support_pos=config["support_config"]["n_support_pos"],
            n_support_neg=config["support_config"]["n_support_neg"],
            n_repeats=config["meta_config"]["n_repeats"],
        )
    except ValueError:
        target_episodes = []

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
    base_model = build_stage2_model_from_config(dummy_processed.shape[1], config, device)
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

    model = build_stage2_model_from_config(dummy_processed.shape[1], config, device)
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
    base_model = build_stage2_model_from_config(dummy_processed.shape[1], config, device)
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
        "synthetic_proportion": syn_cfg.get("synthetic_proportion", 0.0),
        "augmentation_method": syn_cfg.get("augmentation_method", "none"),
        "target_per_class": syn_cfg.get("target_per_class"),
        "k_neighbors": syn_cfg.get(
            "k_neighbors", DEFAULT_SYNTHETIC_CONFIG["k_neighbors"]
        ),
        "status": result.get("status"),
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
    save_stage2_result(result, config, output_dir, experiment_id=experiment_id)

    return {
        "prepared": prepared,
        "result": result,
        "summary": summarize_stage2_result(result, config, experiment_id=experiment_id),
    }
