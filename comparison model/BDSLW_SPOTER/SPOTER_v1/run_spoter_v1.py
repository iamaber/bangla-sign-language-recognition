#!/usr/bin/env python3
"""Simple script to run SPOTER v1 baseline training."""

import sys
import subprocess


def run_spoter_v1():
    """Run SPOTER v1 baseline training."""
    # Set WandB API key
    api_key = "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"

    # Change to SPOTER v1 directory
    import os

    os.chdir("comparison model/BDSLW_SPOTER/SPOTER_v1")

    # Run training
    result = subprocess.run(
        [
            sys.executable,  # Get Python executable
            "train.py",
            "data/train_data.npz",  # Use existing train data
            "data/val_data.npz",  # Use existing val data
            "--epochs",
            "40",
            "--batch-size",
            "64",
            "--lr",
            "3e-4",
            "--device",
            "cuda",
            "--num-classes",
            "77",
            "--run-name",
            "spoter_v1_baseline",
            "--wandb-project",
            "BB3lAowfaCGkIlsby",
            "--wandb-entity",
            "aber-islam-dev-jvai",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    print("=== SPOTER v1 Baseline Training Complete ===")
    print(result.stdout)

    if result.returncode != 0:
        print(f"ERROR: Training failed with return code {result.returncode}")
        if result.stderr:
            print(result.stderr)
    else:
        print("✅ Training completed successfully!")
        print(
            f"📊 Check WandB dashboard: https://wandb.ai/aber-islam-dev-jvai/BB3lAowfaCGkIlsby"
        )
        print(f"🎯 Experiment: spoter_v1_baseline")


if __name__ == "__main__":
    run_spoter_v1()
