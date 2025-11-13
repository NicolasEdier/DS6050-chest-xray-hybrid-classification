"""
Training script for chest X-ray classification models
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import time
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset import get_dataloaders
from models.baseline import get_model
from models.hybrid import get_hybrid_model
from training.losses import CombinedLoss, get_pos_weights, compute_correlation_matrix
from training.metrics import compute_metrics, AverageMeter


class Trainer:
    """Trainer class for model training and validation"""
    
    def __init__(self,
                 model,
                 train_loader,
                 val_loader,
                 criterion,
                 optimizer,
                 scheduler,
                 device,
                 num_classes=14,
                 save_dir='./checkpoints',
                 log_interval=10):
        """
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            num_classes: Number of disease classes
            save_dir: Directory to save checkpoints
            log_interval: Interval for logging training progress
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_classes = num_classes
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        
        self.best_auroc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_aurocs = []
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        losses = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        
        for batch_idx, (images, labels, _) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            logits = self.model(images)
            
            # Compute loss
            if isinstance(self.criterion, CombinedLoss):
                loss, loss_dict = self.criterion(logits, labels)
            else:
                loss = self.criterion(logits, labels)
                loss_dict = {'total': loss.item()}
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            losses.update(loss.item(), images.size(0))
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        return losses.avg
    
    def validate(self, epoch):
        """Validate model"""
        self.model.eval()
        
        losses = AverageMeter()
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
            
            for images, labels, _ in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits = self.model(images)
                
                # Compute loss
                if isinstance(self.criterion, CombinedLoss):
                    loss, _ = self.criterion(logits, labels)
                else:
                    loss = self.criterion(logits, labels)
                
                # Update metrics
                losses.update(loss.item(), images.size(0))
                
                # Store predictions and labels
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())
                
                pbar.set_postfix({'loss': f'{losses.avg:.4f}'})
        
        # Compute metrics
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        metrics = compute_metrics(all_logits, all_labels)
        
        print(f"\nValidation Results:")
        print(f"  Loss: {losses.avg:.4f}")
        print(f"  AUROC: {metrics['auroc_mean']:.4f}")
        print(f"  AUPRC: {metrics['auprc_mean']:.4f}")
        
        return losses.avg, metrics
    
    def train(self, num_epochs, early_stopping_patience=15):
        """
        Train model for multiple epochs
        
        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Stop if no improvement for this many epochs
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, metrics = self.validate(epoch)
            self.val_losses.append(val_loss)
            self.val_aurocs.append(metrics['auroc_mean'])
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            epoch_time = time.time() - epoch_start
            
            print(f"\nEpoch {epoch}/{num_epochs} - Time: {epoch_time:.1f}s")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val AUROC: {metrics['auroc_mean']:.4f}")
            
            # Save checkpoint
            is_best = metrics['auroc_mean'] > self.best_auroc
            if is_best:
                self.best_auroc = metrics['auroc_mean']
                patience_counter = 0
                print(f"  New best AUROC: {self.best_auroc:.4f}")
                
                # Save best model
                self.save_checkpoint(
                    epoch=epoch,
                    metrics=metrics,
                    is_best=True
                )
            else:
                patience_counter += 1
            
            # Save regular checkpoint
            if epoch % 5 == 0:
                self.save_checkpoint(
                    epoch=epoch,
                    metrics=metrics,
                    is_best=False
                )
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                print(f"Best AUROC: {self.best_auroc:.4f}")
                break
            
            print("-" * 80)
        
        print("\nTraining complete!")
        print(f"Best AUROC: {self.best_auroc:.4f}")
        
        # Save training history
        self.save_history()
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_auroc': self.best_auroc,
            'metrics': metrics,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_aurocs': self.val_aurocs
        }
        
        if is_best:
            path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, path)
            print(f"  Saved best model to {path}")
        else:
            path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, path)
    
    def save_history(self):
        """Save training history"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_aurocs': self.val_aurocs,
            'best_auroc': self.best_auroc
        }
        
        path = self.save_dir / 'training_history.json'
        with open(path, 'w') as f:
            json.dump(history, f, indent=2)


def train_baseline_model(
    model_name='resnet50',
    data_dir='./processed_data',
    image_dir='./NIH_ChestXray',
    batch_size=16,
    num_epochs=100,
    learning_rate=1e-4,
    weight_decay=1e-4,
    num_workers=4,
    device=None,
    save_dir=None
):
    """
    Train a baseline model
    
    Args:
        model_name: 'resnet50', 'densenet121', or 'resnet50_multiscale'
        data_dir: Directory with train/val/test CSV files
        image_dir: Directory with chest X-ray images
        batch_size: Batch size
        num_epochs: Number of epochs
        learning_rate: Initial learning rate
        weight_decay: Weight decay
        num_workers: Number of data loading workers
        device: Device to use (None for auto-detect)
        save_dir: Directory to save checkpoints
    """
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup save directory
    if save_dir is None:
        save_dir = f'./checkpoints/{model_name}'
    
    # Get data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=data_dir,
        image_dir=image_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Create model
    print(f"Creating {model_name} model...")
    model = get_model(model_name=model_name, num_classes=14, pretrained=True)
    model = model.to(device)
    
    # Compute class weights
    print("Computing class weights...")
    pos_weights = get_pos_weights(train_loader, device)
    print(f"Positive class weights: {pos_weights}")
    
    # Create loss function
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Create scheduler (warmup + cosine annealing)
    warmup_epochs = 5
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=warmup_epochs * len(train_loader)
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=(num_epochs - warmup_epochs) * len(train_loader)
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs * len(train_loader)]
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=save_dir
    )
    
    # Train
    trainer.train(num_epochs=num_epochs, early_stopping_patience=15)
    
    return trainer


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train chest X-ray classification model')
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=['resnet50', 'densenet121', 'resnet50_multiscale'],
                       help='Model architecture')
    parser.add_argument('--data_dir', type=str, default='./processed_data',
                       help='Directory with processed data')
    parser.add_argument('--image_dir', type=str, default='./NIH_ChestXray',
                       help='Directory with images')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    args = parser.parse_args()
    
    train_baseline_model(
        model_name=args.model,
        data_dir=args.data_dir,
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        num_workers=args.num_workers
    )