"""
Evaluate previously trained hybrid models (A0, A3, A4, A5)
WITHOUT retraining.

Usage:
    python run_hybrid_evaluation.py --models all
    python run_hybrid_evaluation.py --models A0 A4
"""

import argparse
import subprocess
from pathlib import Path
import sys


def print_header(text):
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80 + "\n")


def run_command(cmd, description):
    print(f"\n{'='*80}")
    print(description)
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True)
        print(f"\n✓ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} - FAILED")
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate previously trained hybrid models."
    )

    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Which models to evaluate: A0 A3 A4 A5 or 'all'",
    )

    args = parser.parse_args()

    # Normalize input
    user_models = [m.upper() for m in args.models]
    all_variants = ["A0", "A3", "A4", "A5"]

    if user_models == ["ALL"]:
        variants = all_variants
    else:
        # Validate model names
        invalid = [m for m in user_models if m not in all_variants]
        if invalid:
            print(f"Invalid model(s): {invalid}. Valid: A0 A3 A4 A5 or ALL")
            sys.exit(1)
        variants = user_models

    print_header("HYBRID ATTENTION NETWORKS - EVALUATION ONLY")

    for variant in variants:
        model_key = f"hybrid_{variant.lower()}"
        checkpoint_path = Path(f"./checkpoints/{model_key}/best_model.pth")

        if not checkpoint_path.exists():
            print(f"\n✗ SKIPPING {variant} — checkpoint not found:")
            print(f"  {checkpoint_path}")
            continue

        # Run evaluation
        success = run_command(
            [
                "python", "evaluation/evaluate.py",
                "--checkpoint", str(checkpoint_path),
                "--model", model_key,
                "--save_dir", "./evaluation_results"
            ],
            f"Evaluating model {variant} from {checkpoint_path}"
        )

        if not success:
            print(f"✗ Evaluation failed for {variant}")

    print_header("EVALUATION COMPLETE")
    print("Results saved in: ./evaluation_results/\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
        sys.exit(1)
