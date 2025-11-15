"""
Evaluation script for trained models
"""

import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('..')

from data.dataset import get_dataloaders
from models.baseline import get_model
from models.hybrid import get_hybrid_model
from training.metrics import (
    compute_metrics, 
    print_metrics_table,
    compute_optimal_thresholds,
    bootstrap_confidence_interval
)


class ModelEvaluator:
    """Evaluator class for model testing"""
    
    def __init__(self, model, test_loader, device, disease_classes=None):
        """
        Args:
            model: Trained PyTorch model
            test_loader: Test data loader
            device: Device to use
            disease_classes: List of disease class names
        """
        self.model = model
        self.test_loader = test_loader
        self.device = device
        
        if disease_classes is None:
            self.disease_classes = [
                'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
                'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
                'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
                'Pleural_Thickening', 'Hernia'
            ]
        else:
            self.disease_classes = disease_classes
    
    def evaluate(self, threshold=0.5, compute_ci=False):
        """
        Evaluate model on test set
        
        Args:
            threshold: Classification threshold
            compute_ci: Whether to compute confidence intervals (slow)
        
        Returns:
            metrics: Dictionary with evaluation metrics
            predictions: Dictionary with logits, probs, and labels
        """
        self.model.eval()
        
        all_logits = []
        all_labels = []
        all_image_names = []
        
        print("Running inference on test set...")
        with torch.no_grad():
            for images, labels, image_names in tqdm(self.test_loader):
                images = images.to(self.device)
                
                # Forward pass
                logits = self.model(images)
                
                # Store results
                all_logits.append(logits.cpu())
                all_labels.append(labels)
                all_image_names.extend(image_names)
        
        # Concatenate results
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Compute metrics
        print("\nComputing metrics...")
        metrics = compute_metrics(all_logits, all_labels, threshold=threshold)
        
        # Print results
        print_metrics_table(metrics, self.disease_classes)
        
        # Compute confidence intervals if requested
        if compute_ci:
            print("\nComputing 95% confidence intervals (this may take a few minutes)...")
            ci = bootstrap_confidence_interval(all_logits, all_labels)
            metrics['confidence_intervals'] = ci
            
            print("\nAUROC with 95% CI:")
            for i, disease in enumerate(self.disease_classes):
                mean = ci['mean'][i]
                lower = ci['lower'][i]
                upper = ci['upper'][i]
                print(f"  {disease:<20}: {mean:.4f} [{lower:.4f}, {upper:.4f}]")
        
        # Store predictions
        predictions = {
            'logits': all_logits.numpy(),
            'labels': all_labels.numpy(),
            'probs': torch.sigmoid(all_logits).numpy(),
            'image_names': all_image_names
        }
        
        return metrics, predictions
    
    def find_optimal_thresholds(self):
        """Find optimal classification thresholds for each disease"""
        print("Finding optimal thresholds...")
        
        # Get predictions
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels, _ in tqdm(self.test_loader):
                images = images.to(self.device)
                logits = self.model(images)
                all_logits.append(logits.cpu())
                all_labels.append(labels)
        
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Compute optimal thresholds
        thresholds = compute_optimal_thresholds(all_logits, all_labels)
        
        print("\nOptimal thresholds:")
        for i, disease in enumerate(self.disease_classes):
            print(f"  {disease:<20}: {thresholds[i]:.3f}")
        
        return thresholds
    
    def analyze_errors(self, predictions, model_name, save_dir=None):
        """
        Analyze false positives and false negatives
        
        Args:
            predictions: Dictionary with logits, labels, probs, image_names
            save_dir: Directory to save analysis results
        """
        print("\nAnalyzing prediction errors...")
        
        probs = predictions['probs']
        labels = predictions['labels']
        image_names = predictions['image_names']
        
        # Use threshold of 0.5
        preds = (probs >= 0.5).astype(int)
        
        error_analysis = {}
        
        for i, disease in enumerate(self.disease_classes):
            # False positives: predicted 1, actual 0
            fp_indices = np.where((preds[:, i] == 1) & (labels[:, i] == 0))[0]
            
            # False negatives: predicted 0, actual 1
            fn_indices = np.where((preds[:, i] == 0) & (labels[:, i] == 1))[0]
            
            # True positives
            tp_indices = np.where((preds[:, i] == 1) & (labels[:, i] == 1))[0]
            
            # True negatives
            tn_indices = np.where((preds[:, i] == 0) & (labels[:, i] == 0))[0]
            
            error_analysis[disease] = {
                'num_fp': len(fp_indices),
                'num_fn': len(fn_indices),
                'num_tp': len(tp_indices),
                'num_tn': len(tn_indices),
                'fp_rate': len(fp_indices) / len(tn_indices) if len(tn_indices) > 0 else 0,
                'fn_rate': len(fn_indices) / len(tp_indices) if len(tp_indices) > 0 else 0,
                'fp_examples': [image_names[idx] for idx in fp_indices[:5]],
                'fn_examples': [image_names[idx] for idx in fn_indices[:5]]
            }
            
            print(f"\n{disease}:")
            print(f"  TP: {len(tp_indices)}, TN: {len(tn_indices)}")
            print(f"  FP: {len(fp_indices)}, FN: {len(fn_indices)}")
            print(f"  FP Rate: {error_analysis[disease]['fp_rate']:.3f}")
            print(f"  FN Rate: {error_analysis[disease]['fn_rate']:.3f}")
        
        # Save error analysis
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            with open(save_dir / f'{model_name}_error_analysis.json', 'w') as f:
                json.dump(error_analysis, f, indent=2)
            print(f"\nError analysis saved to {save_dir / 'error_analysis.json'}")
        
        return error_analysis


def load_checkpoint(checkpoint_path, model):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Best AUROC: {checkpoint['best_auroc']:.4f}")
    return model


def evaluate_model(
    checkpoint_path,
    model_name='resnet50',
    data_dir='./processed_data',
    image_dir='./NIH_ChestXray',
    batch_size=32,
    num_workers=4,
    save_dir='./evaluation_results',
    compute_ci=False
):
    """
    Evaluate a trained model
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_name: Model architecture name
        data_dir: Directory with data CSV files
        image_dir: Directory with images
        batch_size: Batch size for evaluation
        num_workers: Number of data loading workers
        save_dir: Directory to save results
        compute_ci: Whether to compute confidence intervals
    """
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading test data...")
    _, _, test_loader = get_dataloaders(
        data_dir=data_dir,
        image_dir=image_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Create model
    print(f"Creating {model_name} model...")
    if 'hybrid' in model_name:
        model = get_hybrid_model(model_name.replace('hybrid_', ''))
    else:
        model = get_model(model_name)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    model = load_checkpoint(checkpoint_path, model)
    model = model.to(device)
    
    # Create evaluator
    evaluator = ModelEvaluator(model, test_loader, device)
    
    # Evaluate
    metrics, predictions = evaluator.evaluate(compute_ci=compute_ci)
    
    # Find optimal thresholds
    optimal_thresholds = evaluator.find_optimal_thresholds()
    
    # Analyze errors
    error_analysis = evaluator.analyze_errors(predictions, model_name, save_dir=save_dir)
    
    # Save results
    results = {
        'model': model_name,
        'checkpoint': str(checkpoint_path),
        'metrics': {
            'auroc_mean': float(metrics['auroc_mean']),
            'auprc_mean': float(metrics['auprc_mean']),
            'f1_mean': float(metrics['f1_mean']),
            'hamming_loss': float(metrics['hamming_loss']),
            'auroc_per_class': [float(x) for x in metrics['auroc_per_class']],
            'auprc_per_class': [float(x) for x in metrics['auprc_per_class']],
            'f1_per_class': [float(x) for x in metrics['f1_per_class']]
        },
        'optimal_thresholds': optimal_thresholds.tolist()
    }
    
    with open(save_dir / f'{model_name}_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save predictions
    np.savez(
        save_dir / f"{model_name}_predictions.npz",
        logits=predictions['logits'],
        labels=predictions['labels'],
        probs=predictions['probs']
    )
    
    print(f"\nResults saved to {save_dir}")
    
    return metrics, predictions


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model', type=str, default='resnet50',
                       help='Model architecture')
    parser.add_argument('--data_dir', type=str, default='./processed_data',
                       help='Directory with data')
    parser.add_argument('--image_dir', type=str, default='./NIH_ChestXray',
                       help='Directory with images')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--save_dir', type=str, default='./evaluation_results',
                       help='Directory to save results')
    parser.add_argument('--compute_ci', action='store_true',
                       help='Compute confidence intervals')
    
    args = parser.parse_args()
    
    evaluate_model(
        checkpoint_path=args.checkpoint,
        model_name=args.model,
        data_dir=args.data_dir,
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        compute_ci=args.compute_ci
    )
