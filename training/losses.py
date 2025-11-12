"""
Loss functions for multi-label classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross Entropy Loss for class imbalance"""
    
    def __init__(self, pos_weights):
        """
        Args:
            pos_weights: Tensor of shape (num_classes,) with positive class weights
        """
        super(WeightedBCELoss, self).__init__()
        self.pos_weights = pos_weights
    
    def forward(self, logits, targets):
        """
        Args:
            logits: Predictions of shape (batch_size, num_classes)
            targets: Ground truth of shape (batch_size, num_classes)
        
        Returns:
            loss: Scalar loss value
        """
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weights
        )
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Reference: https://arxiv.org/abs/1708.02002
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Weighting factor (0-1)
            gamma: Focusing parameter (typically 2)
            reduction: 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        Args:
            logits: Predictions of shape (batch_size, num_classes)
            targets: Ground truth of shape (batch_size, num_classes)
        
        Returns:
            loss: Scalar loss value
        """
        # Get probabilities
        probs = torch.sigmoid(logits)
        
        # Compute focal loss components
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        # BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        # Apply focal weight and alpha
        focal_loss = self.alpha * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CorrelationAwareLoss(nn.Module):
    """
    Correlation-aware loss that penalizes inconsistent predictions
    for frequently co-occurring diseases
    """
    
    def __init__(self, correlation_matrix, threshold=0.3, lambda_corr=0.1):
        """
        Args:
            correlation_matrix: Tensor of shape (num_classes, num_classes)
                               with disease correlation values
            threshold: Minimum correlation to consider
            lambda_corr: Weight for correlation loss term
        """
        super(CorrelationAwareLoss, self).__init__()
        self.register_buffer('correlation_matrix', correlation_matrix)
        self.threshold = threshold
        self.lambda_corr = lambda_corr
        
        # Create mask for correlated pairs
        mask = (correlation_matrix > threshold).float()
        # Zero out diagonal
        mask.fill_diagonal_(0)
        self.register_buffer('corr_mask', mask)
    
    def forward(self, logits, targets=None):
        """
        Args:
            logits: Predictions of shape (batch_size, num_classes)
            targets: Not used, kept for API consistency
        
        Returns:
            loss: Correlation penalty
        """
        # Get probabilities
        probs = torch.sigmoid(logits)
        
        batch_size, num_classes = probs.shape
        
        # Compute pairwise differences
        # prob_diff[i,j] = |p_i - p_j| for each sample
        probs_expanded1 = probs.unsqueeze(2)  # (B, C, 1)
        probs_expanded2 = probs.unsqueeze(1)  # (B, 1, C)
        prob_diff = torch.abs(probs_expanded1 - probs_expanded2)  # (B, C, C)
        
        # Weight by correlation and mask
        weighted_diff = prob_diff * self.corr_mask.unsqueeze(0)
        weighted_diff = weighted_diff * self.correlation_matrix.unsqueeze(0)
        
        # Average over batch and pairs
        loss = weighted_diff.sum() / (batch_size * self.corr_mask.sum())
        
        return self.lambda_corr * loss


class CombinedLoss(nn.Module):
    """
    Combined loss function with BCE, Focal Loss, and Correlation-Aware components
    """
    
    def __init__(self,
                 pos_weights=None,
                 correlation_matrix=None,
                 use_focal=True,
                 use_correlation=True,
                 focal_alpha=0.25,
                 focal_gamma=2.0,
                 lambda_focal=0.5,
                 lambda_corr=0.1):
        """
        Args:
            pos_weights: Positive class weights for BCE
            correlation_matrix: Disease correlation matrix
            use_focal: Whether to include focal loss
            use_correlation: Whether to include correlation loss
            focal_alpha: Focal loss alpha parameter
            focal_gamma: Focal loss gamma parameter
            lambda_focal: Weight for focal loss
            lambda_corr: Weight for correlation loss
        """
        super(CombinedLoss, self).__init__()
        
        # Weighted BCE
        self.bce_loss = WeightedBCELoss(pos_weights) if pos_weights is not None else nn.BCEWithLogitsLoss()
        
        # Focal loss
        self.use_focal = use_focal
        if use_focal:
            self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            self.lambda_focal = lambda_focal
        
        # Correlation loss
        self.use_correlation = use_correlation
        if use_correlation and correlation_matrix is not None:
            self.corr_loss = CorrelationAwareLoss(
                correlation_matrix=correlation_matrix,
                lambda_corr=lambda_corr
            )
    
    def forward(self, logits, targets):
        """
        Args:
            logits: Predictions of shape (batch_size, num_classes)
            targets: Ground truth of shape (batch_size, num_classes)
        
        Returns:
            loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        # BCE loss
        loss_bce = self.bce_loss(logits, targets)
        total_loss = loss_bce
        
        loss_dict = {'bce': loss_bce.item()}
        
        # Focal loss
        if self.use_focal:
            loss_focal = self.focal_loss(logits, targets)
            total_loss = total_loss + self.lambda_focal * loss_focal
            loss_dict['focal'] = loss_focal.item()
        
        # Correlation loss
        if self.use_correlation:
            loss_corr = self.corr_loss(logits)
            total_loss = total_loss + loss_corr
            loss_dict['correlation'] = loss_corr.item()
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


def compute_correlation_matrix(train_loader, device, num_classes=14):
    """
    Compute correlation matrix from training data
    
    Args:
        train_loader: Training data loader
        device: Device to use
        num_classes: Number of disease classes
    
    Returns:
        correlation_matrix: Tensor of shape (num_classes, num_classes)
    """
    # Collect all labels
    all_labels = []
    for _, labels, _ in train_loader:
        all_labels.append(labels)
    
    all_labels = torch.cat(all_labels, dim=0).to(device)  # (N, C)
    
    # Compute correlation matrix
    # Center the data
    mean = all_labels.mean(dim=0, keepdim=True)
    centered = all_labels - mean
    
    # Compute correlation
    cov = (centered.T @ centered) / (all_labels.size(0) - 1)
    std = all_labels.std(dim=0, keepdim=True)
    correlation = cov / (std.T @ std + 1e-8)
    
    # Ensure values are in [-1, 1]
    correlation = torch.clamp(correlation, -1, 1)
    
    return correlation


def get_pos_weights(train_loader, device, num_classes=14):
    """
    Compute positive class weights from training data
    
    Args:
        train_loader: Training data loader
        device: Device to use
        num_classes: Number of disease classes
    
    Returns:
        pos_weights: Tensor of shape (num_classes,)
    """
    # Count positive samples for each class
    pos_counts = torch.zeros(num_classes)
    total_samples = 0
    
    for _, labels, _ in train_loader:
        pos_counts += labels.sum(dim=0)
        total_samples += labels.size(0)
    
    # Compute weights (inverse frequency)
    neg_counts = total_samples - pos_counts
    pos_weights = neg_counts / (pos_counts + 1e-8)
    
    # Normalize
    pos_weights = pos_weights / pos_weights.mean()
    
    return pos_weights.to(device)


if __name__ == '__main__':
    # Test loss functions
    batch_size = 4
    num_classes = 14
    
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, 2, (batch_size, num_classes)).float()
    
    # Test weighted BCE
    pos_weights = torch.ones(num_classes)
    criterion = WeightedBCELoss(pos_weights)
    loss = criterion(logits, targets)
    print(f"Weighted BCE Loss: {loss.item():.4f}")
    
    # Test focal loss
    criterion = FocalLoss()
    loss = criterion(logits, targets)
    print(f"Focal Loss: {loss.item():.4f}")
    
    # Test correlation loss
    corr_matrix = torch.rand(num_classes, num_classes)
    corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make symmetric
    criterion = CorrelationAwareLoss(corr_matrix)
    loss = criterion(logits)
    print(f"Correlation Loss: {loss.item():.4f}")
    
    # Test combined loss
    criterion = CombinedLoss(
        pos_weights=pos_weights,
        correlation_matrix=corr_matrix,
        use_focal=True,
        use_correlation=True
    )
    loss, loss_dict = criterion(logits, targets)
    print(f"\nCombined Loss: {loss.item():.4f}")
    print("Loss components:", loss_dict)
