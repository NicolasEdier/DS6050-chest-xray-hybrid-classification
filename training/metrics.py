"""
Evaluation metrics for multi-label classification
"""

import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_recall_curve,
    roc_curve
)


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_metrics(logits, labels, threshold=0.5):
    """
    Compute evaluation metrics for multi-label classification
    
    Args:
        logits: Predicted logits of shape (N, num_classes)
        labels: Ground truth labels of shape (N, num_classes)
        threshold: Threshold for binary predictions
    
    Returns:
        metrics: Dictionary with various metrics
    """
    # Convert to numpy
    if torch.is_tensor(logits):
        logits = logits.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    
    # Get probabilities
    probs = 1 / (1 + np.exp(-logits))  # Sigmoid
    
    # Binary predictions
    preds = (probs >= threshold).astype(int)
    
    num_classes = labels.shape[1]
    
    # Per-class AUROC
    auroc_per_class = []
    for i in range(num_classes):
        if len(np.unique(labels[:, i])) > 1:  # Need both classes present
            auroc = roc_auc_score(labels[:, i], probs[:, i])
            auroc_per_class.append(auroc)
        else:
            auroc_per_class.append(np.nan)
    
    # Per-class AUPRC (Average Precision)
    auprc_per_class = []
    for i in range(num_classes):
        if len(np.unique(labels[:, i])) > 1:
            auprc = average_precision_score(labels[:, i], probs[:, i])
            auprc_per_class.append(auprc)
        else:
            auprc_per_class.append(np.nan)
    
    # Per-class F1 score
    f1_per_class = []
    for i in range(num_classes):
        if labels[:, i].sum() > 0:  # At least one positive sample
            f1 = f1_score(labels[:, i], preds[:, i], zero_division=0)
            f1_per_class.append(f1)
        else:
            f1_per_class.append(np.nan)
    
    # Overall metrics
    metrics = {
        'auroc_per_class': auroc_per_class,
        'auroc_mean': np.nanmean(auroc_per_class),
        'auprc_per_class': auprc_per_class,
        'auprc_mean': np.nanmean(auprc_per_class),
        'f1_per_class': f1_per_class,
        'f1_mean': np.nanmean(f1_per_class),
        'hamming_loss': hamming_loss(labels, preds)
    }
    
    return metrics


def compute_optimal_thresholds(logits, labels):
    """
    Compute optimal thresholds for each class based on F1 score
    
    Args:
        logits: Predicted logits of shape (N, num_classes)
        labels: Ground truth labels of shape (N, num_classes)
    
    Returns:
        thresholds: Optimal threshold for each class
    """
    if torch.is_tensor(logits):
        logits = logits.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    
    probs = 1 / (1 + np.exp(-logits))
    num_classes = labels.shape[1]
    
    thresholds = []
    for i in range(num_classes):
        if labels[:, i].sum() == 0:
            thresholds.append(0.5)
            continue
        
        # Compute precision-recall curve
        precision, recall, thresh = precision_recall_curve(labels[:, i], probs[:, i])
        
        # Compute F1 for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # Find optimal threshold
        best_idx = np.argmax(f1_scores)
        if best_idx < len(thresh):
            thresholds.append(thresh[best_idx])
        else:
            thresholds.append(0.5)
    
    return np.array(thresholds)


def print_metrics_table(metrics, disease_classes=None):
    """
    Print metrics in a formatted table
    
    Args:
        metrics: Dictionary with metrics from compute_metrics
        disease_classes: List of disease class names
    """
    if disease_classes is None:
        disease_classes = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
            'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
            'Pleural_Thickening', 'Hernia'
        ]
    
    print("\nPer-Class Metrics:")
    print("-" * 70)
    print(f"{'Disease':<20} {'AUROC':>10} {'AUPRC':>10} {'F1':>10}")
    print("-" * 70)
    
    for i, disease in enumerate(disease_classes):
        auroc = metrics['auroc_per_class'][i]
        auprc = metrics['auprc_per_class'][i]
        f1 = metrics['f1_per_class'][i]
        
        print(f"{disease:<20} {auroc:>10.4f} {auprc:>10.4f} {f1:>10.4f}")
    
    print("-" * 70)
    print(f"{'Average':<20} {metrics['auroc_mean']:>10.4f} {metrics['auprc_mean']:>10.4f} {metrics['f1_mean']:>10.4f}")
    print("-" * 70)
    print(f"Hamming Loss: {metrics['hamming_loss']:.4f}")


def compute_confusion_matrix_metrics(labels, preds):
    """
    Compute confusion matrix metrics (TP, FP, TN, FN) for each class
    
    Args:
        labels: Ground truth labels (N, num_classes)
        preds: Predicted labels (N, num_classes)
    
    Returns:
        metrics_dict: Dictionary with TP, FP, TN, FN for each class
    """
    num_classes = labels.shape[1]
    
    metrics_dict = {
        'tp': [],
        'fp': [],
        'tn': [],
        'fn': [],
        'sensitivity': [],
        'specificity': [],
        'precision': []
    }
    
    for i in range(num_classes):
        tp = np.sum((labels[:, i] == 1) & (preds[:, i] == 1))
        fp = np.sum((labels[:, i] == 0) & (preds[:, i] == 1))
        tn = np.sum((labels[:, i] == 0) & (preds[:, i] == 0))
        fn = np.sum((labels[:, i] == 1) & (preds[:, i] == 0))
        
        metrics_dict['tp'].append(tp)
        metrics_dict['fp'].append(fp)
        metrics_dict['tn'].append(tn)
        metrics_dict['fn'].append(fn)
        
        # Sensitivity (Recall)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics_dict['sensitivity'].append(sensitivity)
        
        # Specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics_dict['specificity'].append(specificity)
        
        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics_dict['precision'].append(precision)
    
    return metrics_dict


def bootstrap_confidence_interval(logits, labels, n_bootstrap=1000, confidence=0.95):
    """
    Compute confidence intervals for AUROC using bootstrap
    
    Args:
        logits: Predicted logits (N, num_classes)
        labels: Ground truth labels (N, num_classes)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95 for 95% CI)
    
    Returns:
        ci_dict: Dictionary with mean, lower, and upper bounds for each class
    """
    if torch.is_tensor(logits):
        logits = logits.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    
    probs = 1 / (1 + np.exp(-logits))
    n_samples, num_classes = labels.shape
    
    alpha = (1 - confidence) / 2
    
    ci_dict = {
        'mean': [],
        'lower': [],
        'upper': []
    }
    
    for i in range(num_classes):
        if len(np.unique(labels[:, i])) <= 1:
            ci_dict['mean'].append(np.nan)
            ci_dict['lower'].append(np.nan)
            ci_dict['upper'].append(np.nan)
            continue
        
        aurocs = []
        for _ in range(n_bootstrap):
            # Sample with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            
            # Ensure both classes present
            if len(np.unique(labels[indices, i])) > 1:
                auroc = roc_auc_score(labels[indices, i], probs[indices, i])
                aurocs.append(auroc)
        
        aurocs = np.array(aurocs)
        ci_dict['mean'].append(np.mean(aurocs))
        ci_dict['lower'].append(np.percentile(aurocs, alpha * 100))
        ci_dict['upper'].append(np.percentile(aurocs, (1 - alpha) * 100))
    
    return ci_dict


if __name__ == '__main__':
    # Test metrics
    num_samples = 100
    num_classes = 14
    
    # Generate synthetic data
    logits = torch.randn(num_samples, num_classes)
    labels = torch.randint(0, 2, (num_samples, num_classes)).float()
    
    # Compute metrics
    metrics = compute_metrics(logits, labels)
    
    print("Testing metrics computation:")
    print_metrics_table(metrics)
    
    # Compute optimal thresholds
    thresholds = compute_optimal_thresholds(logits, labels)
    print(f"\nOptimal thresholds: {thresholds}")
    
    # Compute confidence intervals
    print("\nComputing confidence intervals (this may take a moment)...")
    ci = bootstrap_confidence_interval(logits, labels, n_bootstrap=100)
    print("AUROC with 95% CI:")
    for i in range(num_classes):
        print(f"  Class {i}: {ci['mean'][i]:.3f} [{ci['lower'][i]:.3f}, {ci['upper'][i]:.3f}]")
