#!/usr/bin/env python3
"""validate_stage2b_accounting.py

Validate Stage 2b experiment folders for support accounting completeness.

Stale vs current distinction
-----------------------------
Folders that pre-date the Phase 6 accounting fix will not have
support_accounting.csv or stage2b_config.json.  The validator separates
these from *current* folders (those written after the fix) so that:

  * Stale folders are reported as WARNings, not FAILures.
  * Current smote_nc folders that are missing support_accounting.csv
    or have n_synthetic_support_rows == 0 are hard FAILures.

A folder is classified as **current** if it contains stage2b_config.json
(written by the fixed save_stage2_result).  Without that file it is stale.

Checks (current folders only):
  1. smote_nc folders must have support_accounting.csv.
  2. Every smote_nc folder must either have n_synthetic_support_rows > 0
     (SMOTENC fired) OR a non-empty gen_skip_reason explaining why augmentation
     was skipped (e.g. "real_support_exceeds_target_effective_k").
     → FAIL: zero synthetic rows AND no skip reason.
     → WARN: zero synthetic rows WITH a documented skip reason.
  3. n_query_synthetic_rows must be 0 in every accounting row (invariant).
  4. gen-none folders must have n_synthetic_support_rows == 0 if accounting exists.
  5. AUROC must differ between matched smote_nc and gen-none pairs.

Run from the project root:
    python experiments/validate_stage2b_accounting.py

Exit code 0 = all checks pass (warnings OK).
Exit code 1 = one or more checks failed.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("pandas is required: pip install pandas --break-system-packages")
    sys.exit(1)

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_STAGE2 = PROJECT_ROOT / "models" / "stage_2" / "experiments"


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}", file=sys.stderr)


def warn(msg: str) -> None:
    print(f"  WARN: {msg}")


def ok(msg: str) -> None:
    print(f"  OK  : {msg}")


def parse_experiment_id(exp_id: str) -> dict:
    """Parse key fields from a Stage 2(b) experiment folder name."""
    info: dict = {"exp_id": exp_id, "is_stage2b": False}
    parts = exp_id.split("__")
    for part in parts:
        if part.startswith("init-"):
            info["init_name"] = part[5:]
        elif part.startswith("method-"):
            info["method"] = part[7:]
        elif part.startswith("kpos-"):
            info["kpos"] = int(part[5:])
        elif part.startswith("kneg-"):
            info["kneg"] = int(part[5:])
        elif part.startswith("steps-"):
            info["steps"] = int(part[6:])
        elif part.startswith("target-"):
            info["target"] = part[7:]
        elif part.startswith("synprop-"):
            info["synprop"] = float(part[8:])
            info["is_stage2b"] = True
        elif part.startswith("gen-"):
            info["gen"] = part[4:]
        elif part.startswith("tek-"):
            info["tek"] = int(part[4:])
    return info


def is_current_folder(d: Path) -> bool:
    """A folder is 'current' (post-fix) iff stage2b_config.json is present."""
    return (d / "stage2b_config.json").exists()


def run_checks() -> int:
    n_fail = 0
    n_warn = 0

    print("=" * 70)
    print("Stage 2b Accounting Validation  (stale-aware)")
    print("=" * 70)

    if not MODELS_STAGE2.is_dir():
        fail(f"Stage 2 experiments directory not found: {MODELS_STAGE2}")
        return 1

    # ── Discover all Stage 2b folders ────────────────────────────────────────
    all_dirs = [d for d in MODELS_STAGE2.iterdir() if d.is_dir()]
    stage2b_dirs = [d for d in all_dirs if "synprop-" in d.name and "gen-" in d.name]

    print(f"\nTotal experiment folders     : {len(all_dirs)}")
    print(f"Stage 2b folders             : {len(stage2b_dirs)}")

    if not stage2b_dirs:
        warn("No Stage 2b experiment folders found (synprop- + gen- in name).")
        warn("Stage 2b experiments may not have been run yet.")
        n_warn += 1
        print("\n" + "=" * 70)
        print("RESULT: 0 FAILURES (no Stage 2b folders to validate)")
        print("=" * 70)
        return 0

    # Separate stale (pre-fix) from current (post-fix)
    current_dirs = [d for d in stage2b_dirs if is_current_folder(d)]
    stale_dirs = [d for d in stage2b_dirs if not is_current_folder(d)]

    print(f"  Current (post-fix)         : {len(current_dirs)}")
    print(f"  Stale  (pre-fix, no config): {len(stale_dirs)}")
    if stale_dirs:
        warn(
            f"{len(stale_dirs)} stale Stage 2b folder(s) lack stage2b_config.json "
            f"— these pre-date the accounting fix and are skipped."
        )
        n_warn += 1

    if not current_dirs:
        warn("No current (post-fix) Stage 2b folders to validate.")
        warn("Re-run experiments with --force-rerun to generate current folders.")
        n_warn += 1
        print("\n" + "=" * 70)
        print(f"RESULT: 0 FAILURES ({n_warn} warning(s) — all folders are stale)")
        print("=" * 70)
        return 0

    parsed_current = [parse_experiment_id(d.name) for d in current_dirs]
    smote_dirs = [(d, p) for d, p in zip(current_dirs, parsed_current)
                  if p.get("gen", "") == "smote_nc"]
    none_dirs = [(d, p) for d, p in zip(current_dirs, parsed_current)
                 if p.get("gen", "") == "none"]

    print(f"\nCurrent smote_nc folders     : {len(smote_dirs)}")
    print(f"Current gen-none folders     : {len(none_dirs)}")

    # ── Check 1: support_accounting.csv must exist for current smote_nc ───────
    print("\n[1] support_accounting.csv existence (current smote_nc folders)")
    missing_acct = []
    for d, p in smote_dirs:
        if not (d / "support_accounting.csv").exists():
            missing_acct.append(d.name)

    if missing_acct:
        fail(
            f"{len(missing_acct)}/{len(smote_dirs)} current smote_nc folders "
            f"missing support_accounting.csv"
        )
        for name in missing_acct[:5]:
            fail(f"  Missing: {name}")
        if len(missing_acct) > 5:
            fail(f"  ... and {len(missing_acct) - 5} more")
        n_fail += 1
    else:
        ok(f"All {len(smote_dirs)} current smote_nc folders have support_accounting.csv ✓")

    # ── Check 2: SMOTENC generated synthetic rows OR has a documented skip ─────
    # A zero-synthetic smote_nc run is acceptable ONLY when gen_skip_reason
    # explains why (e.g. real support already meets target_effective_k).
    # Sources checked in order: per-episode CSV column → stage2b_config.json.
    print("\n[2] SMOTENC generated synthetic rows (or documented skip reason)")
    zero_syn_no_reason: list[str] = []  # FAIL: zero synthetic, no reason
    zero_syn_with_reason: list[tuple[str, str]] = []  # WARN: zero but documented
    n_fired = 0

    def _get_skip_reason_from_config(folder: Path) -> str:
        """Read summary-level gen_skip_reason from stage2b_config.json."""
        import json
        cfg_path = folder / "stage2b_config.json"
        if not cfg_path.exists():
            return ""
        try:
            with open(cfg_path) as _f:
                cfg = json.load(_f)
            return cfg.get("gen_skip_reason") or ""
        except Exception:
            return ""

    for d, p in smote_dirs:
        acct_path = d / "support_accounting.csv"
        if not acct_path.exists():
            continue
        try:
            acct = pd.read_csv(acct_path)
        except Exception as exc:
            fail(f"{d.name}/support_accounting.csv: cannot read ({exc})")
            n_fail += 1
            continue

        if "n_synthetic_support_rows" not in acct.columns:
            warn(f"{d.name}/support_accounting.csv: n_synthetic_support_rows column missing")
            n_warn += 1
            continue

        if (acct["n_synthetic_support_rows"] == 0).all():
            # Determine skip reason: check per-episode column first, then config JSON.
            skip_reason = ""
            if "gen_skip_reason" in acct.columns:
                reasons = acct["gen_skip_reason"].dropna().unique()
                reasons = [r for r in reasons if r]
                skip_reason = "|".join(sorted(set(reasons))) if reasons else ""
            if not skip_reason:
                skip_reason = _get_skip_reason_from_config(d)

            if skip_reason:
                zero_syn_with_reason.append((d.name, skip_reason))
            else:
                zero_syn_no_reason.append(d.name)
        else:
            mean_syn = acct["n_synthetic_support_rows"].mean()
            n_fired += 1

    # Report results
    if n_fired > 0:
        ok(f"{n_fired} smote_nc folder(s) generated synthetic rows ✓")
    if zero_syn_with_reason:
        for name, reason in zero_syn_with_reason:
            warn(
                f"{name}: n_synthetic_support_rows=0 — WARN (documented skip: {reason})"
            )
        n_warn += len(zero_syn_with_reason)
    if zero_syn_no_reason:
        for name in zero_syn_no_reason:
            fail(
                f"{name}: n_synthetic_support_rows=0 in ALL episodes "
                f"AND no gen_skip_reason — SMOTENC silently did nothing!"
            )
        n_fail += 1

    zero_syn_count = len(zero_syn_no_reason) + len(zero_syn_with_reason)
    if zero_syn_count == 0 and not missing_acct and smote_dirs:
        ok("SMOTENC generated synthetic rows in all current smote_nc folders ✓")
    elif zero_syn_with_reason and not zero_syn_no_reason:
        ok(f"All zero-synthetic smote_nc runs have documented skip reasons (WARN, not FAIL) ✓")

    # ── Check 3: n_query_synthetic_rows must be 0 (invariant) ────────────────
    print("\n[3] n_query_synthetic_rows == 0 invariant")
    query_syn_violations = 0
    for d, p in smote_dirs + none_dirs:
        acct_path = d / "support_accounting.csv"
        if not acct_path.exists():
            continue
        try:
            acct = pd.read_csv(acct_path)
        except Exception:
            continue
        if "n_query_synthetic_rows" in acct.columns:
            if (acct["n_query_synthetic_rows"] > 0).any():
                query_syn_violations += 1
                fail(
                    f"{d.name}: n_query_synthetic_rows > 0 "
                    f"— synthetic rows must never appear in the query set!"
                )
                n_fail += 1

    if query_syn_violations == 0:
        ok("n_query_synthetic_rows == 0 in all current folders ✓")

    # ── Check 4: gen-none folders have n_synthetic_support_rows == 0 ─────────
    print("\n[4] Current gen-none folders have zero synthetic support rows")
    nonzero_syn_none = 0
    for d, p in none_dirs:
        acct_path = d / "support_accounting.csv"
        if not acct_path.exists():
            continue
        try:
            acct = pd.read_csv(acct_path)
        except Exception:
            continue
        if "n_synthetic_support_rows" in acct.columns:
            if (acct["n_synthetic_support_rows"] > 0).any():
                nonzero_syn_none += 1
                fail(
                    f"{d.name}: n_synthetic_support_rows > 0 in a gen-none experiment"
                )
                n_fail += 1

    if nonzero_syn_none == 0:
        ok("No synthetic rows found in gen-none experiments ✓")

    # ── Check 5: AUROC divergence between current smote_nc and none ──────────
    print("\n[5] AUROC divergence between current smote_nc and gen-none")
    summary_path = MODELS_STAGE2 / "experiment_summary.csv"
    if not summary_path.exists():
        warn(f"experiment_summary.csv not found — skipping AUROC check")
        n_warn += 1
    else:
        try:
            summary = pd.read_csv(summary_path)
            # Restrict to IDs that are in current post-fix folders
            current_ids = {d.name for d in current_dirs}
            s2b = summary[
                summary["experiment_id"].isin(current_ids)
                & summary["augmentation_method"].notna()
                & (summary.get("status", pd.Series("success", index=summary.index)) == "success")
            ].copy()

            if s2b.empty:
                warn(
                    "No successful current Stage 2b rows in experiment_summary.csv — "
                    "rerun experiments and re-register results to populate it"
                )
                n_warn += 1
            else:
                smote_rows = s2b[s2b["augmentation_method"] == "smote_nc"].copy()
                none_rows = s2b[s2b["augmentation_method"] == "none"].copy()

                key_cols = ["method", "init_name", "target_department",
                            "n_support_pos", "n_support_neg", "inner_steps"]
                key_cols = [c for c in key_cols if c in s2b.columns]

                auroc_col = next(
                    (c for c in ("gold_auroc_mean", "auroc_mean") if c in s2b.columns),
                    None,
                )

                if auroc_col and key_cols and not smote_rows.empty and not none_rows.empty:
                    merged = smote_rows.merge(
                        none_rows[key_cols + [auroc_col]],
                        on=key_cols, suffixes=("_smote", "_none"),
                    )
                    if merged.empty:
                        warn("No matching current smote_nc / none pairs for AUROC comparison")
                        n_warn += 1
                    else:
                        col_smote = f"{auroc_col}_smote"
                        col_none = f"{auroc_col}_none"
                        merged["auroc_diff"] = (
                            merged[col_smote] - merged[col_none]
                        ).abs()
                        identical = merged[merged["auroc_diff"] < 1e-6]
                        if len(identical) > 0:
                            fail(
                                f"{len(identical)} current smote_nc experiments have AUROC "
                                f"identical to gen-none (within 1e-6) — SMOTENC may not be applied"
                            )
                            for _, row in identical.head(5).iterrows():
                                fail(
                                    f"  {row.get('target_department','?')} | "
                                    f"{row.get('method','?')} | "
                                    f"k={row.get('n_support_pos','?')} | "
                                    f"AUROC={row[col_smote]:.4f}"
                                )
                            n_fail += 1
                        else:
                            ok(
                                f"All {len(merged)} current smote_nc/none pairs have "
                                f"diverging AUROC ✓"
                            )
                else:
                    warn("Insufficient columns or data for AUROC divergence check")
                    n_warn += 1
        except Exception as exc:
            warn(f"AUROC divergence check failed: {exc}")
            n_warn += 1

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    if n_fail == 0 and n_warn == 0:
        print("RESULT: ALL CHECKS PASSED ✓")
    elif n_fail == 0:
        print(f"RESULT: PASSED with {n_warn} warning(s)")
    else:
        print(f"RESULT: {n_fail} CHECK(S) FAILED ✗  ({n_warn} warning(s))")
    print("=" * 70)

    return n_fail


if __name__ == "__main__":
    failures = run_checks()
    sys.exit(0 if failures == 0 else 1)
