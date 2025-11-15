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


def plot_training_history(history_file, save_dir='./plots', model_name='model'):
    import json

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load training history
    with open(history_file, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_losses']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # ----- Training & Validation Loss -----
    ax1.plot(epochs, history['train_losses'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_losses'], 'r-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} — Training & Validation Loss')
    ax1.grid(alpha=0.3)
    ax1.legend()   # FIXED — legend now shows both lines

    # ----- Validation AUROC -----
    ax2.plot(epochs, history['val_aurocs'], 'g-', label='Val AUROC')
    ax2.axhline(history['best_auroc'], color='r', linestyle='--',
                label=f'Best AUROC = {history["best_auroc"]:.4f}')  # FIXED
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('AUROC')
    ax2.set_title(f'{model_name} — Validation AUROC')
    ax2.grid(alpha=0.3)
    ax2.legend()  # FIXED — legend includes both AUROC curve and best line

    plt.tight_layout()

    out_path = save_dir / f'{model_name}_training_history.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Training history plot saved to {out_path}")




def plot_confusion_matrices(predictions_file, disease_classes, save_dir='./plots', model_name='model'):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(predictions_file)
    probs = data['probs']
    labels = data['labels']
    preds = (probs > 0.5).astype(int)

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()

    for i in range(min(len(disease_classes), 16)):
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(labels[:, i], preds[:, i])
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm_norm, annot=True, fmt='.2f',
                    cmap='Blues', ax=axes[i],
                    xticklabels=['Neg', 'Pos'],
                    yticklabels=['Neg', 'Pos'])

        axes[i].set_title(f'{model_name} — {disease_classes[i]}')

    plt.suptitle(f'{model_name} — Confusion Matrices', fontsize=18)
    plt.tight_layout()

    out_path = save_dir / f'{model_name}_confusion_matrices.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Confusion matrices saved to {out_path}")



def plot_roc_curves(predictions_file, disease_classes, save_dir='./plots', model_name='model'):
    from sklearn.metrics import roc_curve, auc

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load predictions
    data = np.load(predictions_file)
    probs = data['probs']
    labels = data['labels']

    # Make grid
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    axes = axes.flatten()

    for i, disease in enumerate(disease_classes):
        fpr, tpr, _ = roc_curve(labels[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)

        ax = axes[i]
        ax.plot(fpr, tpr, 'b-', linewidth=2,
                label=f'AUC = {roc_auc:.3f}')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{disease}')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(alpha=0.3)

    # Hide unused panels if disease_classes < 16
    for j in range(len(disease_classes), 16):
        axes[j].axis('off')

    # Super-title for entire grid
    plt.suptitle(f'{model_name} — ROC Curves for All Disease Classes', fontsize=18)

    # ADD EXTRA SPACE to prevent overlap
    plt.subplots_adjust(top=0.92)  # FIX: avoids overlapping with subplot titles

    # Save
    out_path = save_dir / f'{model_name}_roc_curves.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"ROC curves saved to {out_path}")

if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Visualize model results')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name to visualize (must exist in all_models.json)')
    parser.add_argument('--results_dir', type=str, default='./evaluation_results',
                        help='Directory containing all_models.json and prediction files')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                        help='Directory to save visualizations')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all_models.json
    master_json_path = results_dir / "all_models.json"
    if not master_json_path.exists():
        raise FileNotFoundError(f"ERROR: {master_json_path} not found")

    with open(master_json_path, 'r') as f:
        master_data = json.load(f)

    model_name = args.model
    if model_name not in master_data:
        raise ValueError(f"Model '{model_name}' not found in all_models.json")

    model_entry = master_data[model_name]

    # Extract paths
    history_path = model_entry.get("training_history_path", None)
    pred_path = results_dir / f"{model_name}_predictions.npz"

    disease_classes = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
        'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
        'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
        'Pleural_Thickening', 'Hernia'
    ]

    # --- Training History ---
    if history_path and Path(history_path).exists():
        plot_training_history(history_path, save_dir=output_dir, model_name=model_name)

    # --- Predictions ---
    if pred_path.exists():
        plot_confusion_matrices(pred_path, disease_classes, save_dir=output_dir, model_name=model_name)
        plot_roc_curves(pred_path, disease_classes, save_dir=output_dir, model_name=model_name)


    print(f"\nAll visualizations saved to {output_dir}")

