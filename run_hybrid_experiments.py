"""
Complete experiment runner for hybrid models
Usage: python run_hybrid_experiments.py
"""

import subprocess
import sys
from pathlib import Path
import json
import time


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80 + "\n")


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} - FAILED")
        print(f"Error: {e}")
        return False


def main():
    print_header("HYBRID ATTENTION NETWORKS - COMPLETE EXPERIMENT PIPELINE")
    
    print("This script will:")
    print("  1. Train all 4 hybrid variants (A0, A3, A4, A5)")
    print("  2. Evaluate each variant on the test set")
    print("  3. Generate visualizations and comparisons")
    print("  4. Analyze gate values for A5")
    print("\nEstimated time: 6-8 hours (25 epochs × 4 variants)")
    
    response = input("\nProceed? (y/n): ").lower()
    if response != 'y':
        print("Aborted.")
        return
    
    start_time = time.time()
    
    # ========================================================================
    # STEP 1: Train all hybrid variants
    # ========================================================================
    
    print_header("STEP 1: TRAINING HYBRID VARIANTS")
    
    success = run_command(
        ['python', 'training/train_hybrid.py', '--variant', 'all', '--epochs', '25'],
        "Training all variants (A0, A3, A4, A5)"
    )
    
    if not success:
        print("\n✗ Training failed. Aborting.")
        return
    
    # ========================================================================
    # STEP 2: Evaluate all variants
    # ========================================================================
    
    print_header("STEP 2: EVALUATING ALL VARIANTS")
    
    variants = ['A0', 'A3', 'A4', 'A5']
    
    for variant in variants:
        checkpoint_path = f'./checkpoints/hybrid_{variant.lower()}/best_model.pth'
        
        if not Path(checkpoint_path).exists():
            print(f"✗ Checkpoint not found for {variant}: {checkpoint_path}")
            continue
        
        success = run_command(
            [
                'python', 'evaluation/evaluate.py',
                '--checkpoint', checkpoint_path,
                '--model', f'hybrid_{variant.lower()}',
                '--save_dir', './evaluation_results'
            ],
            f"Evaluating variant {variant}"
        )
        
        if not success:
            print(f"✗ Evaluation failed for {variant}")
    
    # ========================================================================
    # STEP 3: Generate visualizations for all variants
    # ========================================================================
    
    print_header("STEP 3: GENERATING VISUALIZATIONS")
    
    for variant in variants:
        model_name = f'hybrid_{variant.lower()}'
        
        success = run_command(
            [
                'python', 'evaluation/visualize.py',
                '--model', model_name,
                '--results_dir', './evaluation_results',
                '--output_dir', f'./visualizations/{model_name}'
            ],
            f"Generating visualizations for {variant}"
        )
    
    # ========================================================================
    # STEP 4: Generate comparison plots
    # ========================================================================
    
    print_header("STEP 4: GENERATING COMPARISON PLOTS")
    
    run_command(
        ['python', 'evaluation/compare_models.py'],
        "Creating model comparison visualizations"
    )
    
    # ========================================================================
    # STEP 5: Analyze A5 gate values
    # ========================================================================
    
    print_header("STEP 5: ANALYZING A5 GATE VALUES")
    
    # Create gate analysis script
    gate_analysis_code = '''
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load A5 training history
history_path = Path("checkpoints/hybrid_a5/training_history.json")
with open(history_path, 'r') as f:
    history = json.load(f)

if 'gate_interpretation' in history:
    gates = history['gate_interpretation']
    
    # Sort by gate value
    sorted_gates = sorted(gates.items(), key=lambda x: x[1], reverse=True)
    
    diseases = [item[0] for item in sorted_gates]
    values = [item[1] for item in sorted_gates]
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    colors = ['#2ecc71' if v > 0.5 else '#e74c3c' for v in values]
    plt.barh(diseases, values, color=colors)
    plt.axvline(x=0.5, color='black', linestyle='--', linewidth=1)
    plt.xlabel('Gate Value (α)', fontsize=12)
    plt.title('A5 Per-Disease Gate Values\\n(α>0.5: CNN-dominant, α<0.5: ViT-dominant)', 
              fontsize=14)
    plt.xlim(0, 1)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    output_path = Path("evaluation/visualizations/a5_gate_analysis.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved gate analysis to {output_path}")
    
    # Print interpretation
    print("\\nGate Value Interpretation:")
    print("-" * 60)
    for disease, value in sorted_gates:
        dominant = "CNN" if value > 0.5 else "ViT"
        strength = abs(value - 0.5) * 2  # 0-1 scale
        print(f"{disease:20s}: {value:.3f} ({dominant}-dominant, strength: {strength:.2f})")
else:
    print("No gate values found in training history")
'''
    
    gate_script_path = Path('analyze_gates.py')
    with open(gate_script_path, 'w') as f:
        f.write(gate_analysis_code)
    
    run_command(
        ['python', 'analyze_gates.py'],
        "Analyzing A5 gate values"
    )
    
    gate_script_path.unlink()  # Clean up temp script
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    
    print_header("EXPERIMENT COMPLETE!")
    
    print(f"Total time: {hours}h {minutes}m")
    print("\nResults locations:")
    print("  - Training logs: ./checkpoints/hybrid_*/")
    print("  - Evaluation results: ./evaluation_results/")
    print("  - Visualizations: ./visualizations/")
    print("  - Model comparisons: ./evaluation/visualizations/")
    
    # Load and display results
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    # Load all_models.json
    all_models_path = Path('evaluation_results/all_models.json')
    if all_models_path.exists():
        with open(all_models_path, 'r') as f:
            results = json.load(f)
        
        print("\nModel Performance Comparison:")
        print("-" * 60)
        
        # Baseline models
        print("\nBaseline Models:")
        for model in ['resnet50', 'densenet121', 'resnet50_multiscale']:
            if model in results:
                auroc = results[model]['metrics']['auroc_mean']
                f1 = results[model]['metrics']['f1_mean']
                print(f"  {model:25s}: AUROC {auroc:.4f}, F1 {f1:.4f}")
        
        # Hybrid models
        print("\nHybrid Models:")
        for variant in ['A0', 'A3', 'A4', 'A5']:
            model_key = f'hybrid_{variant.lower()}'
            if model_key in results:
                auroc = results[model_key]['metrics']['auroc_mean']
                f1 = results[model_key]['metrics']['f1_mean']
                print(f"  {model_key:25s}: AUROC {auroc:.4f}, F1 {f1:.4f}")
        
        # Calculate improvements
        if 'resnet50' in results and 'hybrid_a5' in results:
            baseline_auroc = results['resnet50']['metrics']['auroc_mean']
            hybrid_auroc = results['hybrid_a5']['metrics']['auroc_mean']
            improvement = hybrid_auroc - baseline_auroc
            improvement_pct = (improvement / baseline_auroc) * 100
            
            print(f"\nA5 vs ResNet-50 Baseline:")
            print(f"  Absolute improvement: +{improvement:.4f}")
            print(f"  Relative improvement: +{improvement_pct:.2f}%")
    
    print("\n" + "="*80)
    print("Next steps:")
    print("  1. Review visualizations in ./visualizations/")
    print("  2. Check error analysis in ./evaluation_results/")
    print("  3. Examine A5 gate values for clinical interpretation")
    print("  4. Prepare final report with key findings")
    print("="*80 + "\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)