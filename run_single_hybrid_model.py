"""
Hybrid model training launcher
Run one hybrid variant at a time.

Usage:
    python run_hybrid_experiments.py --model A4 --epochs 25
    python run_hybrid_experiments.py --model A0
    python run_hybrid_experiments.py --model all
"""

import argparse
import subprocess
import sys
import time


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
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} FAILED")
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Train a single hybrid model variant")
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["A0", "A3", "A4", "A5", "all"],
        help="Which hybrid variant to train",
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of epochs to train",
    )
    
    args = parser.parse_args()
    model = args.model.upper()
    epochs = args.epochs

    print_header(f"TRAINING HYBRID MODEL: {model}")

    start_time = time.time()

    # Mapping: if user picks "all", train all 4 variants
    variants = ["A0", "A3", "A4", "A5"] if model == "ALL" else [model]

    for variant in variants:
        print_header(f"TRAINING VARIANT {variant}")

        cmd = [
            "python",
            "training/train_hybrid.py",
            "--variant", variant,
            "--epochs", str(epochs)
        ]

        success = run_command(cmd, f"Training variant {variant}")
        
        if not success:
            print(f"\n✗ Training aborted for {variant}")
            break

    elapsed = time.time() - start_time
    minutes = int((elapsed % 3600) // 60)
    hours = int(elapsed // 3600)

    print_header("TRAINING COMPLETE")
    print(f"Total time: {hours}h {minutes}m\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(1)
