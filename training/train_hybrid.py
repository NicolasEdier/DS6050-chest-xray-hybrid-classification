"""
Training script for hybrid CNN-Transformer models (A0, A3, A4, A5)
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import time
from pathlib import Path
import json
from tqdm import tqdm
import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset import get_dataloaders
from models.hybrid import get_hybrid_model
from models.baseline import count_parameters
from training.losses import get_pos_weights
from training.metrics import compute_metrics, AverageMeter


class HybridTrainer:
    """Trainer for hybrid models"""
    
    def __init__(self,
                 model,
                 train_loader,
                 val_loader,
                 criterion,
                 optimizer,
                 scheduler,
                 device,
                 variant='A0',
                 num_classes=14,
                 save_dir='./checkpoints',
                 log_interval=10):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.variant = variant
        self.num_classes = num_classes
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        
        self.best_auroc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_aurocs = []
        
        # For A5: track gate values during validation
        self.gate_history = [] if variant == 'A5' else None
    
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
            loss = self.criterion(logits, labels)
            
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
        epoch_gates = [] if self.variant == 'A5' else None
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
            
            for images, labels, _ in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                if self.variant == 'A5':
                    logits, gates = self.model(images, return_gates=True)
                    # Average gate values across batch
                    batch_gate_avg = torch.stack([g.mean() for g in gates])
                    epoch_gates.append(batch_gate_avg)
                else:
                    logits = self.model(images)
                
                # Compute loss
                loss = self.criterion(logits, labels)
                
                # Update metrics
                losses.update(loss.item(), images.size(0))
                
                # Store predictions
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())
                
                pbar.set_postfix({'loss': f'{losses.avg:.4f}'})
        
        # Compute metrics
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        metrics = compute_metrics(all_logits, all_labels)
        
        # Store gate values for A5
        if self.variant == 'A5' and epoch_gates:
            avg_gates = torch.stack(epoch_gates).mean(dim=0)
            self.gate_history.append(avg_gates.numpy())
        
        print(f"\nValidation Results:")
        print(f"  Loss: {losses.avg:.4f}")
        print(f"  AUROC: {metrics['auroc_mean']:.4f}")
        print(f"  AUPRC: {metrics['auprc_mean']:.4f}")
        
        return losses.avg, metrics
    
    def train(self, num_epochs, early_stopping_patience=15):
        """Train model for multiple epochs"""
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Variant: {self.variant}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {count_parameters(self.model):,}")
        
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
                self.save_checkpoint(epoch, metrics, is_best=True)
            else:
                patience_counter += 1
            
            # Save regular checkpoint
            if epoch % 5 == 0:
                self.save_checkpoint(epoch, metrics, is_best=False)
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
            
            print("-" * 80)
        
        print(f"\n✓ Training complete! Best AUROC: {self.best_auroc:.4f}")
        self.save_history()
        
        return self.best_auroc
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'variant': self.variant,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_auroc': self.best_auroc,
            'metrics': metrics,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_aurocs': self.val_aurocs
        }
        
        # Add gate history for A5
        if self.variant == 'A5' and self.gate_history:
            checkpoint['gate_history'] = self.gate_history
        
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
            'variant': self.variant,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_aurocs': self.val_aurocs,
            'best_auroc': self.best_auroc
        }
        
        # Add final gate values for A5
        if self.variant == 'A5' and self.gate_history:
            import numpy as np
            final_gates = np.array(self.gate_history[-1])
            history['final_gate_values'] = final_gates.tolist()
            
            # Disease names for interpretation
            disease_classes = [
                'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
                'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
                'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
                'Pleural_Thickening', 'Hernia'
            ]
            history['gate_interpretation'] = {
                disease: float(gate_val)
                for disease, gate_val in zip(disease_classes, final_gates)
            }
        
        path = self.save_dir / 'training_history.json'
        with open(path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"Saved training history to {path}")


def train_hybrid_model(
    variant='A0',
    data_dir='./processed_data',
    image_dir='./NIH_ChestXray',
    batch_size=16,
    num_epochs=25,
    learning_rate=1e-4,
    weight_decay=1e-4,
    num_workers=4,
    device=None,
    save_dir=None
):
    """
    Train a hybrid model variant
    
    Args:
        variant: 'A0', 'A3', 'A4', or 'A5'
        data_dir: Directory with data
        image_dir: Directory with images
        batch_size: Batch size (recommend 8 for hybrid models)
        num_epochs: Number of epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        num_workers: Data loading workers
        device: Device to use
        save_dir: Checkpoint directory
    """
    # Setup
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if save_dir is None:
        save_dir = f'./checkpoints/hybrid_{variant.lower()}'
    
    print(f"\n{'='*80}")
    print(f"TRAINING HYBRID MODEL - VARIANT {variant}")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Save directory: {save_dir}")
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=data_dir,
        image_dir=image_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Create model
    print(f"\nCreating {variant} model...")
    model = get_hybrid_model(variant=variant, num_classes=14, pretrained=True)
    model = model.to(device)
    print(f"Parameters: {count_parameters(model):,}")
    
    # Compute class weights
    print("\nComputing class weights...")
    pos_weights = get_pos_weights(train_loader, device)
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Scheduler (warmup + cosine)
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
    trainer = HybridTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        variant=variant,
        save_dir=save_dir
    )
    
    # Train
    best_auroc = trainer.train(
        num_epochs=num_epochs,
        early_stopping_patience=15
    )
    
    return trainer, best_auroc


def train_all_variants(
    data_dir='./processed_data',
    image_dir='./NIH_ChestXray',
    batch_size=16,
    num_epochs=25,
    device=None
):
    """Train all hybrid variants sequentially"""
    
    variants = ['A0', 'A3', 'A4', 'A5']
    results = {}
    
    print("\n" + "="*80)
    print("TRAINING ALL HYBRID VARIANTS")
    print("="*80)
    print(f"Variants to train: {', '.join(variants)}")
    print(f"Epochs per variant: {num_epochs}")
    print("="*80)
    
    for variant in variants:
        print(f"\n\n{'='*80}")
        print(f"STARTING VARIANT {variant}")
        print(f"{'='*80}\n")
        
        try:
            trainer, best_auroc = train_hybrid_model(
                variant=variant,
                data_dir=data_dir,
                image_dir=image_dir,
                batch_size=batch_size,
                num_epochs=num_epochs,
                device=device
            )
            
            results[variant] = {
                'best_auroc': best_auroc,
                'status': 'completed'
            }
            
            print(f"\n✓ {variant} completed with AUROC: {best_auroc:.4f}")
            
        except Exception as e:
            print(f"\n✗ Error training {variant}: {e}")
            results[variant] = {
                'status': 'failed',
                'error': str(e)
            }
            continue
    
    # Save overall results
    results_path = Path('./checkpoints/hybrid_training_summary.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("ALL VARIANTS TRAINING COMPLETE")
    print("="*80)
    print("\nResults Summary:")
    for variant, result in results.items():
        if result['status'] == 'completed':
            print(f"  {variant}: AUROC {result['best_auroc']:.4f} ✓")
        else:
            print(f"  {variant}: FAILED ✗")
    
    print(f"\nResults saved to: {results_path}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train hybrid models')
    parser.add_argument('--variant', type=str, default='all',
                       choices=['A0', 'A3', 'A4', 'A5', 'all'],
                       help='Which variant to train')
    parser.add_argument('--data_dir', type=str, default='./processed_data')
    parser.add_argument('--image_dir', type=str, default='./NIH_ChestXray')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size (8-16 recommended for hybrid)')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=1e-4)
    
    args = parser.parse_args()
    
    if args.variant == 'all':
        train_all_variants(
            data_dir=args.data_dir,
            image_dir=args.image_dir,
            batch_size=args.batch_size,
            num_epochs=args.epochs
        )
    else:
        train_hybrid_model(
            variant=args.variant,
            data_dir=args.data_dir,
            image_dir=args.image_dir,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr
        )