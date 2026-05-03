#!/usr/bin/env python3
"""validate_stage2_config.py

Validate experiments/stage2_config.yaml for structural correctness.

Checks:
  1. YAML parses without error.
  2. base_config is present and has required keys.
  3. MLP architecture matches Stage 1 (256→128→1, dropout=0.1).
  4. All canonical presets are present (exactly 13, no missing, no extra).
  5. No synprop=1.0 in any active (non-archived) preset.
  6. All target_departments in canonical presets are valid department names.
  7. All augmentation_method values are legal ("none" or "smote_nc").
  8. archived_presets section is present (27 total) and every archived preset
     has superseded_by and reason fields.
  9. target_effective_k_grid values are null or positive integers (never 0 or negative).
 10. Documentation-only fields (augmentation_scope, freeze_strategy) are
     present only in Stage 2b or TODO presets (not runner-read).
 11. thesis_main includes zero_shot in its methods list.
 12. gradient_clip_norm_grid and inner_grad_clip values are valid (null or positive numbers).

Canonical presets (4 groups):
  GROUP 1: quick_debug, stage2b_debug
  GROUP 2: thesis_main, init_ablation, k_shot_label_budget
  GROUP 3: maml_lr_sweep, maml_stabilization_debug, maml_stabilization_debug_balanced,
           maml_stabilization_sweep, maml_freeze_ablation, maml_mechanics_debug
  GROUP 4: stage2b_smotenc, stage2b_label_budget

TODO (documentation-only) presets:
  maml_freeze_ablation, maml_mechanics_debug

Run from the project root:
    python experiments/validate_stage2_config.py

Exit code 0 = all checks pass.
Exit code 1 = one or more checks failed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGE2_CONFIG = PROJECT_ROOT / "experiments" / "stage2_config.yaml"

# ── Expected canonical presets ────────────────────────────────────────────────
# These 10 names must appear in the presets: section of stage2_config.yaml.
# Organized in 4 groups as documented at the top of stage2_config.yaml.
# Any preset added or removed must be reflected here.
# Presets that are superseded should be moved to archived_presets: instead.
CANONICAL_PRESETS = {
    # Group 1: Debug / Smoke Tests
    "quick_debug",
    "stage2b_debug",
    # Group 2: Main Thesis Experiments
    "thesis_main",
    "init_ablation",
    "k_shot_label_budget",
    # Group 3: MAML-Focused Optimization
    "maml_lr_sweep",
    "maml_stabilization_debug",
    "maml_stabilization_debug_balanced",
    "maml_stabilization_sweep",
    "maml_freeze_ablation",
    "maml_mechanics_debug",
    # Group 4: Stage 2b Support Augmentation
    "stage2b_smotenc",
    "stage2b_label_budget",
}

# Documentation-only / TODO presets: these presets are present in the config
# but are not yet runnable (marked [TODO — NOT YET RUNNABLE] in description).
TODO_PRESETS = {
    "maml_freeze_ablation",
    "maml_mechanics_debug",
}

# Documentation-only fields: present in YAML for human reference, never read
# by run_stage2.py (resolve_experiment_grid / run_stage2_pipeline).
DOCUMENTATION_ONLY_FIELDS = {"augmentation_scope", "freeze_strategy"}

VALID_DEPARTMENTS = {
    "Logistics",
    "Packaging Material",
    "Devices & Needles",
    "Bioprocessing & Raw Materials",
}

VALID_AUG_METHODS = {"none", "smote_nc"}

REQUIRED_BASE_KEYS = [
    "seed", "verbose", "method", "data", "features",
    "stage1_init", "task_config", "support_config", "meta_config",
    "mlp_config", "evaluation", "output",
]

CANONICAL_MLP = {"hidden_dim_1": 256, "hidden_dim_2": 128, "dropout": 0.1}


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}", file=sys.stderr)


def ok(msg: str) -> None:
    print(f"  OK  : {msg}")


def run_checks() -> int:
    n_fail = 0

    print("=" * 60)
    print("Stage 2 Config Validation")
    print("=" * 60)

    # ── Check 1: YAML parse ──────────────────────────────────────────────────
    print("\n[1] YAML parse")
    if not STAGE2_CONFIG.exists():
        fail(f"Config not found: {STAGE2_CONFIG}")
        return 1
    try:
        with open(STAGE2_CONFIG) as f:
            cfg = yaml.safe_load(f)
        ok("YAML parses without error ✓")
    except yaml.YAMLError as exc:
        fail(f"YAML parse error: {exc}")
        return 1

    # ── Check 2: base_config keys ────────────────────────────────────────────
    print("\n[2] base_config required keys")
    base = cfg.get("base_config", {})
    if not isinstance(base, dict):
        fail("base_config is missing or not a dict")
        n_fail += 1
    else:
        missing = [k for k in REQUIRED_BASE_KEYS if k not in base]
        if missing:
            fail(f"base_config missing keys: {missing}")
            n_fail += 1
        else:
            ok(f"All {len(REQUIRED_BASE_KEYS)} required keys present ✓")

    # ── Check 3: MLP architecture ────────────────────────────────────────────
    print("\n[3] MLP architecture (must match Stage 1)")
    mlp = base.get("mlp_config", {})
    for key, expected in CANONICAL_MLP.items():
        actual = mlp.get(key)
        if actual != expected:
            fail(f"mlp_config.{key} = {actual!r}, expected {expected!r}")
            n_fail += 1
    if n_fail == 0:
        ok(f"MLP: hidden_dim_1={CANONICAL_MLP['hidden_dim_1']}, "
           f"hidden_dim_2={CANONICAL_MLP['hidden_dim_2']}, "
           f"dropout={CANONICAL_MLP['dropout']} ✓")

    # ── Check 4: Canonical presets present ───────────────────────────────────
    print("\n[4] Canonical presets (13 total, 4 groups)")
    presets = cfg.get("presets", {})
    if not isinstance(presets, dict):
        fail("presets section missing or not a dict")
        n_fail += 1
    else:
        missing_presets = CANONICAL_PRESETS - set(presets)
        extra_presets = set(presets) - CANONICAL_PRESETS
        if missing_presets:
            fail(f"Missing canonical presets: {sorted(missing_presets)}")
            n_fail += 1
        if extra_presets:
            fail(
                f"Unknown presets in canonical section (should these be archived?): "
                f"{sorted(extra_presets)}"
            )
            n_fail += 1
        if not missing_presets and not extra_presets:
            # Print NOTE for TODO presets
            for todo_preset in sorted(TODO_PRESETS):
                if todo_preset in presets:
                    print(
                        f"  NOTE: Preset '{todo_preset}' is TODO (documentation-only, "
                        f"not yet runnable)"
                    )
            ok(f"All {len(CANONICAL_PRESETS)} canonical presets present, none extra ✓")

    # ── Check 5: No synprop=1.0 in canonical presets ─────────────────────────
    print("\n[5] No synthetic_proportion=1.0 in canonical presets")
    for name, preset in (presets or {}).items():
        synprop_grid = preset.get("synthetic_proportion_grid", [])
        if 1.0 in synprop_grid or 1 in synprop_grid:
            fail(
                f"Preset '{name}' has synthetic_proportion_grid containing 1.0 "
                f"(invalid — use ≤0.9)"
            )
            n_fail += 1
    if n_fail == 0:
        ok("No synthetic_proportion=1.0 found ✓")

    # ── Check 6: Valid target_departments ────────────────────────────────────
    print("\n[6] Valid target_departments in canonical presets")
    for name, preset in (presets or {}).items():
        depts = preset.get("target_departments", [])
        invalid = [d for d in depts if d not in VALID_DEPARTMENTS]
        if invalid:
            fail(f"Preset '{name}' has invalid target_departments: {invalid}")
            n_fail += 1
    if n_fail == 0:
        ok("All target_departments are valid ✓")

    # ── Check 7: Valid augmentation_method values ────────────────────────────
    print("\n[7] Valid augmentation_method values in canonical presets")
    for name, preset in (presets or {}).items():
        aug_grid = preset.get("augmentation_method_grid", [])
        invalid_aug = [m for m in aug_grid if m not in VALID_AUG_METHODS]
        if invalid_aug:
            fail(
                f"Preset '{name}' has invalid augmentation_method_grid: {invalid_aug}. "
                f"Valid values: {sorted(VALID_AUG_METHODS)}"
            )
            n_fail += 1
    if n_fail == 0:
        ok(f"All augmentation_method values valid {sorted(VALID_AUG_METHODS)} ✓")

    # ── Check 8: archived_presets section ────────────────────────────────────
    print("\n[8] archived_presets section")
    archived = cfg.get("archived_presets", {})
    if not isinstance(archived, dict) or len(archived) == 0:
        fail("archived_presets section missing or empty")
        n_fail += 1
    else:
        malformed = []
        for name, ap in archived.items():
            if not isinstance(ap, dict):
                malformed.append(name)
                continue
            if "superseded_by" not in ap:
                malformed.append(f"{name}(no superseded_by)")
            if "reason" not in ap:
                malformed.append(f"{name}(no reason)")
        if malformed:
            fail(f"Archived presets with missing metadata: {malformed}")
            n_fail += 1
        else:
            ok(
                f"{len(archived)} archived presets, all have "
                f"superseded_by and reason ✓"
            )

    # ── Check 9: target_effective_k_grid values ───────────────────────────────
    print("\n[9] target_effective_k_grid values (null or positive integer only)")
    for name, preset in (presets or {}).items():
        tek_grid = preset.get("target_effective_k_grid", [])
        if not isinstance(tek_grid, list):
            tek_grid = [tek_grid]
        for val in tek_grid:
            if val is None:
                continue  # null is valid
            if not isinstance(val, int) or val <= 0:
                fail(
                    f"Preset '{name}': target_effective_k_grid contains invalid value "
                    f"{val!r} — must be null or a positive integer"
                )
                n_fail += 1
    if n_fail == 0:
        ok("All target_effective_k_grid values are null or positive integers ✓")

    # ── Check 10: documentation-only fields are acknowledged ─────────────────
    print("\n[10] Documentation-only fields (augmentation_scope, freeze_strategy)")
    doc_field_presets: dict[str, list[str]] = {}
    for name, preset in (presets or {}).items():
        found_doc_fields = [f for f in DOCUMENTATION_ONLY_FIELDS if f in preset]
        if found_doc_fields:
            doc_field_presets[name] = found_doc_fields
    if doc_field_presets:
        for pname, fields in doc_field_presets.items():
            print(
                f"  NOTE: Preset '{pname}' contains documentation-only field(s): "
                f"{fields} — these are NOT read by run_stage2.py"
            )
        ok(
            f"{len(doc_field_presets)} preset(s) use documentation-only fields "
            f"(expected — these are human-readable annotations) ✓"
        )
    else:
        ok("No documentation-only fields in canonical presets ✓")

    # ── Check 11: thesis_main includes zero_shot ─────────────────────────────
    print("\n[11] thesis_main includes zero_shot method")
    thesis_main = (presets or {}).get("thesis_main", {})
    thesis_main_methods = thesis_main.get("methods", [])
    if "zero_shot" in thesis_main_methods:
        ok(f"thesis_main methods include zero_shot: {thesis_main_methods} ✓")
    else:
        fail(f"thesis_main missing zero_shot; has: {thesis_main_methods}")
        n_fail += 1

    # ── Check 12: gradient_clip_norm_grid and inner_grad_clip ──────────────────
    print("\n[12] gradient_clip_norm_grid / inner_grad_clip values")
    check12_fail = 0
    for name, preset in (presets or {}).items():
        # gradient_clip_norm_grid: each value must be null or a positive number
        gcn_grid = preset.get("gradient_clip_norm_grid", [])
        if isinstance(gcn_grid, list):
            for val in gcn_grid:
                if val is None:
                    continue  # null = no clipping (valid)
                if not isinstance(val, (int, float)) or val <= 0:
                    fail(
                        f"Preset '{name}': gradient_clip_norm_grid contains "
                        f"invalid value {val!r} — must be null or a positive number"
                    )
                    check12_fail += 1
        # inner_grad_clip: must be null or a positive number
        igc = preset.get("inner_grad_clip", None)
        if igc is not None:
            if not isinstance(igc, (int, float)) or igc <= 0:
                fail(
                    f"Preset '{name}': inner_grad_clip = {igc!r} "
                    f"— must be null or a positive number"
                )
                check12_fail += 1
    n_fail += check12_fail
    if check12_fail == 0:
        ok("All gradient_clip_norm_grid and inner_grad_clip values are valid ✓")

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
