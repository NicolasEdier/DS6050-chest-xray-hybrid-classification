"""
Visualization tools for model interpretation and results
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import cv2

import sys
sys.path.append('..')


class GradCAM:
    """Gradient-weighted Class Activation Mapping"""
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: PyTorch model
            target_layer: Layer to visualize (e.g., model.layer4)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Hook to save activations"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Hook to save gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, class_idx):
        """
        Generate CAM for specific class
        
        Args:
            input_image: Input tensor (1, 3, H, W)
            class_idx: Class index to visualize
        
        Returns:
            cam: Class activation map (H, W)
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        
        # Backward pass for target class
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Weight activations by gradients (global average pooling)
        weights = gradients.mean(dim=(1, 2), keepdim=True)  # (C, 1, 1)
        
        # Weighted combination
        cam = (weights * activations).sum(dim=0)  # (H, W)
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy()


def overlay_heatmap(image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlay heatmap on image
    
    Args:
        image: Original image (H, W, 3) in range [0, 255]
        heatmap: Heatmap (H, W) in range [0, 1]
        alpha: Transparency of heatmap
        colormap: OpenCV colormap
    
    Returns:
        overlaid: Image with heatmap overlay (H, W, 3)
    """
    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8),
        colormap
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlaid = (1 - alpha) * image + alpha * heatmap_colored
    overlaid = overlaid.astype(np.uint8)
    
    return overlaid


def visualize_predictions(
    model,
    dataloader,
    device,
    disease_classes,
    num_examples=10,
    save_dir='./visualizations'
):
    """
    Visualize model predictions with images
    
    Args:
        model: Trained model
        dataloader: Data loader
        device: Device
        disease_classes: List of disease names
        num_examples: Number of examples to visualize
        save_dir: Directory to save visualizations
    """
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    images_shown = 0
    
    with torch.no_grad():
        for images, labels, image_names in dataloader:
            if images_shown >= num_examples:
                break
            
            images_gpu = images.to(device)
            logits = model(images_gpu)
            probs = torch.sigmoid(logits)
            
            # Process each image in batch
            for i in range(images.size(0)):
                if images_shown >= num_examples:
                    break
                
                # Get image and predictions
                img = images[i]
                true_labels = labels[i].numpy()
                pred_probs = probs[i].cpu().numpy()
                pred_labels = (pred_probs > 0.5).astype(int)
                
                # Denormalize image
                img = img.numpy().transpose(1, 2, 0)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = img * std + mean
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
                
                # Create visualization
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Show image
                ax1.imshow(img)
                ax1.axis('off')
                ax1.set_title(f'Image: {image_names[i]}')
                
                # Show predictions
                y_pos = np.arange(len(disease_classes))
                ax2.barh(y_pos, pred_probs, alpha=0.6, label='Predicted')
                ax2.scatter(true_labels, y_pos, color='red', s=100, 
                           marker='o', label='Ground Truth', zorder=3)
                ax2.set_yticks(y_pos)
                ax2.set_yticklabels(disease_classes, fontsize=8)
                ax2.set_xlabel('Probability')
                ax2.set_title('Predictions vs Ground Truth')
                ax2.legend()
                ax2.grid(axis='x', alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(save_dir / f'prediction_{images_shown:03d}.png', 
                           dpi=150, bbox_inches='tight')
                plt.close()
                
                images_shown += 1
    
    print(f"Saved {images_shown} prediction visualizations to {save_dir}")


def visualize_attention_maps(
    model,
    dataloader,
    device,
    disease_classes,
    target_layer_name='layer4',
    num_examples=5,
    save_dir='./attention_maps'
):
    """
    Generate and visualize Grad-CAM attention maps
    
    Args:
        model: Trained model
        dataloader: Data loader
        device: Device
        disease_classes: List of disease names
        target_layer_name: Name of layer to visualize
        num_examples: Number of examples
        save_dir: Directory to save maps
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Get target layer
    if hasattr(model, target_layer_name):
        target_layer = getattr(model, target_layer_name)
    else:
        print(f"Model doesn't have layer {target_layer_name}")
        return
    
    # Create Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    
    images_shown = 0
    
    for images, labels, image_names in dataloader:
        if images_shown >= num_examples:
            break
        
        # Process each image
        for i in range(images.size(0)):
            if images_shown >= num_examples:
                break
            
            img_tensor = images[i:i+1].to(device)
            true_labels = labels[i].numpy()
            
            # Denormalize image for visualization
            img = images[i].numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img * std + mean
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            
            # Get positive disease indices
            positive_diseases = np.where(true_labels == 1)[0]
            
            if len(positive_diseases) == 0:
                continue
            
            # Generate CAM for each positive disease
            num_diseases = len(positive_diseases)
            fig, axes = plt.subplots(2, num_diseases, 
                                    figsize=(5*num_diseases, 10))
            if num_diseases == 1:
                axes = axes.reshape(2, 1)
            
            for j, disease_idx in enumerate(positive_diseases):
                # Generate CAM
                cam = grad_cam.generate_cam(img_tensor, disease_idx)
                
                # Overlay on image
                overlaid = overlay_heatmap(img, cam)
                
                # Plot original
                axes[0, j].imshow(img)
                axes[0, j].axis('off')
                axes[0, j].set_title(disease_classes[disease_idx])
                
                # Plot with heatmap
                axes[1, j].imshow(overlaid)
                axes[1, j].axis('off')
                axes[1, j].set_title('Attention Map')
            
            plt.suptitle(f'Image: {image_names[i]}', fontsize=14)
            plt.tight_layout()
            plt.savefig(save_dir / f'attention_{images_shown:03d}.png',
                       dpi=150, bbox_inches='tight')
            plt.close()
            
            images_shown += 1
    
    print(f"Saved {images_shown} attention maps to {save_dir}")


def plot_training_history(history_file, save_dir='./plots'):
    """
    Plot training history curves
    
    Args:
        history_file: Path to training_history.json
        save_dir: Directory to save plots
    """
    import json
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load history
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # Plot losses
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(epochs, history['train_losses'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_losses'], 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.plot(epochs, history['val_aurocs'], 'g-', label='Val AUROC')
    ax2.axhline(y=history['best_auroc'], color='r', linestyle='--', 
                label=f'Best AUROC: {history["best_auroc"]:.4f}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('AUROC')
    ax2.set_title('Validation AUROC')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training history plot saved to {save_dir}")


def plot_confusion_matrices(predictions_file, disease_classes, save_dir='./plots'):
    """
    Plot confusion matrices for each disease
    
    Args:
        predictions_file: Path to predictions.npz
        disease_classes: List of disease names
        save_dir: Directory to save plots
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load predictions
    data = np.load(predictions_file)
    probs = data['probs']
    labels = data['labels']
    
    # Binary predictions
    preds = (probs > 0.5).astype(int)
    
    # Plot confusion matrix for each disease
    num_diseases = len(disease_classes)
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()
    
    for i in range(min(num_diseases, 16)):
        # Compute confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(labels[:, i], preds[:, i])
        
        # Normalize
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'],
                   ax=axes[i], cbar=False)
        axes[i].set_title(disease_classes[i])
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    # Hide extra subplots
    for i in range(num_diseases, 16):
        axes[i].axis('off')
    
    plt.suptitle('Confusion Matrices (Normalized)', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrices saved to {save_dir}")


def plot_roc_curves(predictions_file, disease_classes, save_dir='./plots'):
    """
    Plot ROC curves for all diseases
    
    Args:
        predictions_file: Path to predictions.npz
        disease_classes: List of disease names
        save_dir: Directory to save plots
    """
    from sklearn.metrics import roc_curve, auc
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load predictions
    data = np.load(predictions_file)
    probs = data['probs']
    labels = data['labels']
    
    # Plot ROC curves
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    axes = axes.flatten()
    
    for i, disease in enumerate(disease_classes):
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(labels[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        
        # Plot
        axes[i].plot(fpr, tpr, 'b-', linewidth=2,
                    label=f'AUC = {roc_auc:.3f}')
        axes[i].plot([0, 1], [0, 1], 'k--', linewidth=1)
        axes[i].set_xlim([0.0, 1.0])
        axes[i].set_ylim([0.0, 1.05])
        axes[i].set_xlabel('False Positive Rate')
        axes[i].set_ylabel('True Positive Rate')
        axes[i].set_title(f'{disease}')
        axes[i].legend(loc='lower right')
        axes[i].grid(alpha=0.3)
    
    plt.suptitle('ROC Curves for All Disease Classes', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_dir / 'roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"ROC curves saved to {save_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize model results')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory with evaluation results')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    disease_classes = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
        'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
        'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
        'Pleural_Thickening', 'Hernia'
    ]
    
    # Plot training history
    history_file = results_dir.parent / 'training_history.json'
    if history_file.exists():
        plot_training_history(history_file, output_dir)
    
    # Plot confusion matrices
    pred_file = results_dir / 'predictions.npz'
    if pred_file.exists():
        plot_confusion_matrices(pred_file, disease_classes, output_dir)
        plot_roc_curves(pred_file, disease_classes, output_dir)
    
    print(f"\nAll visualizations saved to {output_dir}")
