"""
run_stage2.py — Execution script for the Stage 2 pipeline.

Responsibilities:
- load stage2_config.yaml
- call Stage 2 pipeline
- print execution status
- rely on stage2.py to save artifacts

"""

import yaml
import argparse
from pathlib import Path

from master_thesis.stage2 import run_stage2_pipeline


def main(config_path: str):
    """
    Clean execution script for the Stage 2 pipeline.
    """

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    print(f"🚀 Running Stage 2 Pipeline with config: {config_path}")

    # ---------------------------------------------------------
    # 1. Load configuration
    # ---------------------------------------------------------

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # ---------------------------------------------------------
    # 2. Run Stage 2 pipeline
    # ---------------------------------------------------------

    result = run_stage2_pipeline(config)

    # ---------------------------------------------------------
    # 3. Print summary
    # ---------------------------------------------------------

    prepared = result["prepared"]

    print("")

    print("Target department:")
    print(prepared["target_department"])

    print("Valid departments:")
    print(len(prepared["valid_departments"]))

    print("Meta-train departments:")
    print(len(prepared["meta_train_departments"]))

    print("")

    print("Output directory:")
    print(prepared["paths"].output_dir)

    print("")

    print("✅ Stage 2 complete.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run Stage 2 Pipeline"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="experiments/stage2_config.yaml",
        help="Path to Stage 2 config file",
    )

    args = parser.parse_args()

    main(args.config)

# How to run: 
# python experiments/run_stage2.py \
   #  --config configs/stage2_config.yaml or python experiments/run_stage2.py
