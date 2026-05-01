#!/usr/bin/env python3
"""validate_stage1_artifacts.py

Validate Stage 1 model artifacts for consistency and correctness.

Checks:
  1. Global A_weak_only artifact directory exists and has all required files.
  2. target_department in stage1_config.yaml is a valid department name.
  3. For each target-aware artifact directory that EXISTS, validate:
       - Required files are present.
       - metadata.json target_department matches the directory's department.
  4. All feature_names.json files that exist contain exactly 151 features
     and no leakage columns (label_source, label_note, label_date).
  5. All existing feature_names.json files have identical feature lists
     (no version skew between conditions).

Note: Only Logistics and Packaging Material have trained B/C/D artifacts.
      Devices & Needles and Bioprocessing & Raw Materials are not yet trained.
      Missing conditions for untrained departments are warned, not failed.

Run from the project root:
    python experiments/validate_stage1_artifacts.py

Exit code 0 = all checks pass.
Exit code 1 = one or more checks failed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_STAGE1 = PROJECT_ROOT / "models" / "stage_1"
STAGE1_CONFIG = PROJECT_ROOT / "experiments" / "stage1_config.yaml"

# ── Expected structure ────────────────────────────────────────────────────────
VALID_DEPARTMENTS = [
    "Logistics",
    "Packaging Material",
    "Devices & Needles",
    "Bioprocessing & Raw Materials",
]

CONDITIONS_B_C_D = ["B_gold_only", "C_hybrid", "D_hybrid_unweighted"]

REQUIRED_FILES = [
    "metadata.json",
    "feature_names.json",
    "mlp_pretrained.pt",
    "mlp_pretrained_preprocessor.joblib",
]

LEAKAGE_COLS = {"label_source", "label_note", "label_date"}
CANONICAL_FEATURE_COUNT = 151


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}", file=sys.stderr)


def warn(msg: str) -> None:
    print(f"  WARN: {msg}")


def ok(msg: str) -> None:
    print(f"  OK  : {msg}")


def load_feature_list(path: Path) -> list[str] | None:
    """Load feature list from feature_names.json.

    Handles two formats:
      - List format  : ["col1", "col2", ...]
      - Dict format  : {"n_features": N, "feature_names": ["col1", ...]}
    """
    try:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "feature_names" in data:
            return data["feature_names"]
        return None
    except Exception:
        return None


def run_checks() -> int:
    """Run all checks. Returns number of failures."""
    n_fail = 0
    reference_features: list[str] | None = None

    print("=" * 60)
    print("Stage 1 Artifact Validation")
    print("=" * 60)

    # ── Check 1: stage1_config.yaml ──────────────────────────────────────────
    print("\n[1] stage1_config.yaml")
    if not STAGE1_CONFIG.exists():
        fail(f"Config not found: {STAGE1_CONFIG}")
        n_fail += 1
    else:
        with open(STAGE1_CONFIG) as f:
            s1_cfg = yaml.safe_load(f)
        target_dept = s1_cfg.get("target_department", "")
        if target_dept not in VALID_DEPARTMENTS:
            fail(
                f"target_department={target_dept!r} not in valid list "
                f"{VALID_DEPARTMENTS}"
            )
            n_fail += 1
        else:
            ok(f"target_department={target_dept!r} ✓")

    # ── Check 2: Global A_weak_only ──────────────────────────────────────────
    print("\n[2] Global A_weak_only artifact")
    global_dir = MODELS_STAGE1 / "global" / "A_weak_only"
    if not global_dir.is_dir():
        fail(f"Directory missing: {global_dir}")
        n_fail += 1
    else:
        ok(f"global/A_weak_only directory present ✓")
        for fname in REQUIRED_FILES:
            if not (global_dir / fname).exists():
                fail(f"global/A_weak_only/{fname} missing")
                n_fail += 1

    # ── Check 3: Target-aware B/C/D artifacts (only for existing directories) -
    print("\n[3] Target-aware B/C/D artifact directories")
    trained_dirs: list[tuple[str, str, Path]] = []  # (dept, cond, path)

    for dept in VALID_DEPARTMENTS:
        for cond in CONDITIONS_B_C_D:
            art_dir = MODELS_STAGE1 / dept / cond
            if art_dir.is_dir():
                trained_dirs.append((dept, cond, art_dir))
            else:
                warn(f"{dept}/{cond} not yet trained (directory absent)")

    if not trained_dirs:
        fail("No target-aware artifact directories found at all")
        n_fail += 1
    else:
        ok(f"Found {len(trained_dirs)} trained target-aware artifact directories")

    # Required files in each existing directory
    for dept, cond, art_dir in trained_dirs:
        for fname in REQUIRED_FILES:
            if not (art_dir / fname).exists():
                fail(f"{dept}/{cond}/{fname} missing")
                n_fail += 1

    # ── Check 4: metadata.json department naming ─────────────────────────────
    print("\n[4] metadata.json target_department naming")
    for dept, cond, art_dir in trained_dirs:
        meta_path = art_dir / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        actual_dept = meta.get("target_department", "")
        if actual_dept != dept:
            fail(
                f"{dept}/{cond}/metadata.json: "
                f"target_department={actual_dept!r}, expected {dept!r}"
            )
            n_fail += 1
        else:
            ok(f"{dept}/{cond}: target_department={actual_dept!r} ✓")

    # ── Check 5: feature_names.json — count, leakage, consistency ────────────
    print("\n[5] feature_names.json — count, leakage, consistency")
    all_feat_paths = (
        [global_dir / "feature_names.json"]
        + [art_dir / "feature_names.json" for _, _, art_dir in trained_dirs]
    )
    feat_ok = True
    for feat_path in all_feat_paths:
        if not feat_path.exists():
            continue
        rel = feat_path.relative_to(MODELS_STAGE1)
        features = load_feature_list(feat_path)
        if features is None:
            fail(f"{rel}: could not parse feature list (unexpected format)")
            n_fail += 1
            feat_ok = False
            continue

        # Count check
        n = len(features)
        if n != CANONICAL_FEATURE_COUNT:
            fail(f"{rel}: {n} features (expected {CANONICAL_FEATURE_COUNT})")
            n_fail += 1
            feat_ok = False

        # Leakage check
        found_leakage = set(features) & LEAKAGE_COLS
        if found_leakage:
            fail(f"{rel}: leakage columns found: {sorted(found_leakage)}")
            n_fail += 1
            feat_ok = False

        # Consistency check
        if reference_features is None:
            reference_features = sorted(features)
        elif sorted(features) != reference_features:
            fail(f"{rel}: feature list differs from reference (version skew)")
            n_fail += 1
            feat_ok = False

    if feat_ok:
        ok(
            f"All {len(all_feat_paths)} feature_names.json files: "
            f"{CANONICAL_FEATURE_COUNT} features, no leakage, all identical ✓"
        )

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if n_fail == 0:
        print("RESULT: ALL CHECKS PASSED ✓")
    else:
        print(f"RESULT: {n_fail} CHECK(S) FAILED ✗")
    print("=" * 60)

    return n_fail


if __name__ == "__main__":
    failures = run_checks()
    sys.exit(0 if failures == 0 else 1)
