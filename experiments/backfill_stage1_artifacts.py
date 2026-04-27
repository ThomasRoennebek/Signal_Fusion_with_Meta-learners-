"""
backfill_stage1_artifacts.py — Patch existing Stage 1 artifact directories.

Purpose
-------
Some Stage 1 artifacts were created before the current save logic was in
place and are therefore missing the following files:

  - feature_names.json  (input feature list required for SHAP and Stage 2)

This script reads the fitted preprocessor from each canonical Stage 1
artifact directory, extracts feature names, and writes feature_names.json
into every directory that is missing it.

It is safe to re-run: existing feature_names.json files are NOT overwritten
unless --force is passed.

Usage
-----
# From the thesis root directory:
python experiments/backfill_stage1_artifacts.py
python experiments/backfill_stage1_artifacts.py --force          # overwrite existing
python experiments/backfill_stage1_artifacts.py --dry-run        # only report, no writes
python experiments/backfill_stage1_artifacts.py --target Engineering  # different target

This script also prints a summary of all canonical artifact directories,
highlighting:
  - canonical artifacts (correct layout)
  - legacy flat artifacts (old layout, should NOT be used by any active code)
  - missing files per directory
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

# Ensure project src is on the path.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import joblib

from master_thesis.config import MODELS_STAGE1
from master_thesis.stage1 import (
    GLOBAL_CONDITIONS,
    backfill_feature_names_for_artifact_dir,
    save_stage1_feature_names,
)


# ---------------------------------------------------------------------------
# Known artifact layout
# ---------------------------------------------------------------------------

CANONICAL_GLOBAL_CONDITIONS = ["A_weak_only"]
CANONICAL_TARGET_CONDITIONS = ["B_gold_only", "C_hybrid", "D_hybrid_unweighted"]
ALL_CONDITIONS = CANONICAL_GLOBAL_CONDITIONS + CANONICAL_TARGET_CONDITIONS

REQUIRED_FILES = [
    "mlp_pretrained.pt",
    "mlp_pretrained_preprocessor.joblib",
    "metadata.json",
    "feature_names.json",
    "tabular_results.csv",
]

LEGACY_FLAT_NAMES = set(ALL_CONDITIONS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_feature_names_from_preprocessor(preprocessor_path: Path) -> list[str] | None:
    """Extract feature_names_in_ from a fitted sklearn preprocessor."""
    if not preprocessor_path.exists():
        return None
    try:
        pp = joblib.load(preprocessor_path)
        if hasattr(pp, "feature_names_in_"):
            return list(pp.feature_names_in_)
        warnings.warn(
            f"Preprocessor at '{preprocessor_path}' does not expose "
            f"feature_names_in_.  It may have been fitted on a numpy array "
            f"rather than a DataFrame.  Cannot extract feature names automatically.",
            UserWarning,
        )
        return None
    except Exception as exc:
        warnings.warn(f"Could not load preprocessor at '{preprocessor_path}': {exc}", UserWarning)
        return None


def _audit_dir(artifact_dir: Path) -> dict:
    """Return a status dict for one artifact directory."""
    status = {
        "dir": str(artifact_dir),
        "exists": artifact_dir.exists(),
        "files": {},
        "missing_files": [],
    }
    if not artifact_dir.exists():
        status["missing_files"] = list(REQUIRED_FILES)
        return status
    for fname in REQUIRED_FILES:
        present = (artifact_dir / fname).exists()
        status["files"][fname] = present
        if not present:
            status["missing_files"].append(fname)
    return status


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill missing Stage 1 artifacts.")
    parser.add_argument(
        "--target-department", type=str, default="Logistics",
        help="Target department used for Stage 1 target-aware conditions.",
    )
    parser.add_argument(
        "--force", action="store_true", default=False,
        help="Overwrite existing feature_names.json files.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Only report status; do not write any files.",
    )
    args = parser.parse_args()

    stage1_root = MODELS_STAGE1
    target = args.target_department

    print(f"\n{'='*70}")
    print(f"Stage 1 Artifact Audit & Backfill")
    print(f"  stage1_root      : {stage1_root}")
    print(f"  target_department: {target}")
    print(f"  dry_run          : {args.dry_run}")
    print(f"  force            : {args.force}")
    print(f"{'='*70}\n")

    # ------------------------------------------------------------------
    # 1. Audit canonical artifact directories
    # ------------------------------------------------------------------
    canonical_dirs: list[tuple[str, Path]] = []

    for cond in CANONICAL_GLOBAL_CONDITIONS:
        d = stage1_root / "global" / cond
        canonical_dirs.append((cond, d))

    for cond in CANONICAL_TARGET_CONDITIONS:
        d = stage1_root / target / cond
        canonical_dirs.append((cond, d))

    print("── Canonical Artifact Directories ──────────────────────────────────")
    all_ok = True
    for cond_name, d in canonical_dirs:
        audit = _audit_dir(d)
        missing = audit["missing_files"]
        status_str = "OK" if not missing else f"MISSING: {missing}"
        print(f"  {cond_name:30s}  {status_str}")
        if missing:
            all_ok = False

    # ------------------------------------------------------------------
    # 2. Detect legacy flat artifact directories
    # ------------------------------------------------------------------
    print("\n── Legacy Flat Directories (should NOT be loaded by any active code) ──")
    legacy_found = False
    for d in sorted(stage1_root.iterdir()):
        if not d.is_dir():
            continue
        if d.name in LEGACY_FLAT_NAMES:
            print(f"  [LEGACY] {d}  ← old flat layout; use canonical path instead")
            legacy_found = True
        elif d.name not in {"global", "experiments"} and d.name != target:
            # Other target-department directories from past runs
            for cond in CANONICAL_TARGET_CONDITIONS:
                sub = d / cond
                if sub.exists():
                    print(f"  [OLD TARGET] {sub}  ← from a previous target={d.name} run")
                    legacy_found = True

    if not legacy_found:
        print("  None found.")

    # ------------------------------------------------------------------
    # 3. Backfill feature_names.json where missing
    # ------------------------------------------------------------------
    if not args.dry_run:
        print("\n── Backfilling feature_names.json ───────────────────────────────────")
        for cond_name, d in canonical_dirs:
            fn_path = d / "feature_names.json"
            if fn_path.exists() and not args.force:
                print(f"  {cond_name:30s}  Already present — skipping (use --force to overwrite)")
                continue
            if not d.exists():
                print(f"  {cond_name:30s}  Directory not found — cannot backfill")
                continue

            preprocessor_path = d / "mlp_pretrained_preprocessor.joblib"
            names = _load_feature_names_from_preprocessor(preprocessor_path)

            if names is None:
                print(
                    f"  {cond_name:30s}  Could not extract feature names from preprocessor. "
                    f"Re-run Stage 1 training to regenerate feature_names.json."
                )
                continue

            save_stage1_feature_names(feature_columns=names, output_dir=d)
            print(
                f"  {cond_name:30s}  Wrote feature_names.json ({len(names)} features)"
            )
    else:
        print("\n[dry-run] No files written.")

    # ------------------------------------------------------------------
    # 4. Final summary
    # ------------------------------------------------------------------
    print("\n── Post-Backfill Status ─────────────────────────────────────────────")
    for cond_name, d in canonical_dirs:
        audit = _audit_dir(d)
        missing = audit["missing_files"]
        status_str = "✓ ALL PRESENT" if not missing else f"STILL MISSING: {missing}"
        print(f"  {cond_name:30s}  {status_str}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
