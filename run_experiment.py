"""
Quick start script for running complete experiments
Usage: python run_experiment.py --mode [preprocess|train|evaluate|all]
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_preprocessing(data_dir, output_dir):
    """Run data preprocessing"""
    print("\n" + "="*80)
    print("STEP 1: DATA PREPROCESSING")
    print("="*80)
    
    from data.preprocess import ChestXrayPreprocessor
    
    preprocessor = ChestXrayPreprocessor(data_dir, output_dir)
    
    print("Loading metadata...")
    df = preprocessor.load_metadata()
    
    print("Creating stratified subset...")
    subset_df = preprocessor.stratified_sampling(df)
    
    print("Splitting data...")
    train_df, val_df, test_df = preprocessor.patient_level_split(subset_df)
    
    preprocessor.print_dataset_statistics(train_df, "Training Set")
    preprocessor.print_dataset_statistics(val_df, "Validation Set")
    preprocessor.print_dataset_statistics(test_df, "Test Set")
    
    preprocessor.save_splits(train_df, val_df, test_df)
    
    class_weights = preprocessor.get_class_weights(train_df)
    import json
    with open(Path(output_dir) / 'class_weights.json', 'w') as f:
        json.dump(class_weights, f, indent=2)
    
    print("\n✓ Preprocessing complete!")
    return True


def run_training(model_name, data_dir, image_dir, batch_size, epochs, lr):
    """Run model training"""
    print("\n" + "="*80)
    print(f"STEP 2: TRAINING {model_name.upper()}")
    print("="*80)
    
    from training.train import train_baseline_model
    
    trainer = train_baseline_model(
        model_name=model_name,
        data_dir=data_dir,
        image_dir=image_dir,
        batch_size=batch_size,
        num_epochs=epochs,
        learning_rate=lr
    )
    
    print(f"\n✓ Training complete! Best AUROC: {trainer.best_auroc:.4f}")
    return trainer


def run_evaluation(checkpoint_path, model_name, data_dir, image_dir, save_dir):
    """Run model evaluation"""
    print("\n" + "="*80)
    print("STEP 3: EVALUATION")
    print("="*80)
    
    from evaluation.evaluate import evaluate_model
    
    metrics, predictions = evaluate_model(
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        data_dir=data_dir,
        image_dir=image_dir,
        save_dir=save_dir,
        compute_ci=False  # Set to True for confidence intervals (slower)
    )
    
    print(f"\n✓ Evaluation complete! Results saved to {save_dir}")
    return metrics, predictions


def main():
    parser = argparse.ArgumentParser(description='Run chest X-ray classification experiments')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='all',
                       choices=['preprocess', 'train', 'evaluate', 'all'],
                       help='Which step to run')
    
    # Data paths
    parser.add_argument('--data_dir', type=str, default='./NIH_ChestXray',
                       help='Raw data directory')
    parser.add_argument('--processed_dir', type=str, default='./processed_data',
                       help='Processed data directory')
    parser.add_argument('--image_dir', type=str, default='./NIH_ChestXray',
                       help='Image directory')
    
    # Training arguments
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=['resnet50', 'densenet121', 'resnet50_multiscale'],
                       help='Model to train')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=25,
                       help='Number of epochs (default: 25 for quick test)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    
    # Evaluation arguments
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint path for evaluation')
    parser.add_argument('--eval_dir', type=str, default='./evaluation_results',
                       help='Evaluation results directory')
    
    args = parser.parse_args()
    
    # Create directories
    Path(args.processed_dir).mkdir(parents=True, exist_ok=True)
    Path(args.eval_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Run selected mode
        if args.mode in ['preprocess', 'all']:
            success = run_preprocessing(args.data_dir, args.processed_dir)
            if not success:
                print("Preprocessing failed!")
                return
        
        if args.mode in ['train', 'all']:
            trainer = run_training(
                model_name=args.model,
                data_dir=args.processed_dir,
                image_dir=args.image_dir,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr
            )
            
            # Get checkpoint path
            checkpoint_path = Path(f'./checkpoints/{args.model}/best_model.pth')
        
        if args.mode in ['evaluate', 'all']:
            # Use provided checkpoint or default from training
            if args.checkpoint:
                checkpoint_path = args.checkpoint
            elif args.mode == 'evaluate':
                checkpoint_path = Path(f'./checkpoints/{args.model}/best_model.pth')
            
            if not Path(checkpoint_path).exists():
                print(f"Error: Checkpoint not found at {checkpoint_path}")
                return
            
            metrics, predictions = run_evaluation(
                checkpoint_path=checkpoint_path,
                model_name=args.model,
                data_dir=args.processed_dir,
                image_dir=args.image_dir,
                save_dir=args.eval_dir
            )
        
        print("\n" + "="*80)
        print("ALL STEPS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nQuick Summary:")
        print(f"  - Data: {args.processed_dir}")
        print(f"  - Model: {args.model}")
        if args.mode in ['train', 'all']:
            print(f"  - Best AUROC: {trainer.best_auroc:.4f}")
        print(f"  - Results: {args.eval_dir}")
        
    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║  Hybrid Attention Networks for Chest X-Ray Classification    ║
    ║  UVA School of Data Science - DS 6050                        ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    main()
