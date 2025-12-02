import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2

from models.hybrid import get_hybrid_model
from data.dataset import get_dataloaders

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

MODEL_NAME = "hybrid_a5"
CHECKPOINT_PATH = "checkpoints/hybrid_a5/best_model.pth"   # <-- update if needed
OUTPUT_DIR = Path("visualizations/gradcam_hybrid_a5")
NUM_EXAMPLES = 25
TARGET_LAYER_NAME = "layer4"   # <-- update depending on your architecture

DISEASE_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------
# HELPER CLASS (Grad-CAM)
# ------------------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, class_idx):
        self.model.eval()

        # Forward pass
        preds = self.model(input_tensor)

        # Backward pass for selected class
        self.model.zero_grad()
        preds[0, class_idx].backward()

        gradients = self.gradients[0]       # (C, H, W)
        activations = self.activations[0]   # (C, H, W)

        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = (weights * activations).sum(dim=0)

        cam = torch.relu(cam)
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam.cpu().numpy()

def overlay_heatmap(image, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_color = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    blended = (1 - alpha) * image + alpha * heatmap_color
    return blended.astype(np.uint8)


# ------------------------------------------------------------
# MAIN GRAD-CAM EXECUTION
# ------------------------------------------------------------
def run_gradcam():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    DATA_DIR = Path("processed_data")
    IMAGE_DIR = Path("NIH_ChestXray/images")

    print(f"Loading model: {MODEL_NAME}")
    model = get_hybrid_model(MODEL_NAME).to(device)

    print(f"Loading checkpoint from {CHECKPOINT_PATH}")
    state = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])


    # Load validation set
    print("Loading validation data...")
    _, val_loader, _ = get_dataloaders(
        data_dir=DATA_DIR,
        image_dir=IMAGE_DIR,
        batch_size=8,
        num_workers=4
    )

    # Locate target layer inside model
    if hasattr(model, TARGET_LAYER_NAME):
        target_layer = getattr(model, TARGET_LAYER_NAME)
    else:
        raise ValueError(f"HybridA5 does not have layer '{TARGET_LAYER_NAME}'. "
                        f"Available: conv1, bn1, layer1, layer2, layer3, layer4")


    grad_cam = GradCAM(model, target_layer)

    shown = 0

    for images, labels, image_names in val_loader:
        if shown >= NUM_EXAMPLES:
            break

        images = images.to(device)

        for idx in range(images.size(0)):
            if shown >= NUM_EXAMPLES:
                break

            image_tensor = images[idx:idx+1]
            label_vec = labels[idx].numpy()

            # Get all positive disease indices
            pos_diseases = np.where(label_vec == 1)[0]
            if len(pos_diseases) == 0:
                continue

            # Denormalize image for display
            img = images[idx].cpu().numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = (img * std + mean)
            img = np.clip(img * 255, 0, 255).astype(np.uint8)

            # Plot results
            fig, axes = plt.subplots(2, len(pos_diseases),
                                    figsize=(5 * len(pos_diseases), 8))

            if len(pos_diseases) == 1:
                axes = axes.reshape(2, 1)

            fig.suptitle(f"Image: {image_names[idx]}", fontsize=16)

            for j, disease_idx in enumerate(pos_diseases):
                # Generate Grad-CAM
                cam = grad_cam.generate_cam(image_tensor, disease_idx)

                # Overlay map
                overlay = overlay_heatmap(img, cam)

                # Original
                axes[0, j].imshow(img)
                axes[0, j].set_title(DISEASE_CLASSES[disease_idx])
                axes[0, j].axis("off")

                # CAM
                axes[1, j].imshow(overlay)
                axes[1, j].set_title("Attention Map")
                axes[1, j].axis("off")

            plt.tight_layout()
            save_path = OUTPUT_DIR / f"gradcam_{shown:03d}.png"
            plt.savefig(save_path, dpi=150)
            plt.close()

            print(f"Saved: {save_path}")
            shown += 1

    print("Done.")


if __name__ == "__main__":
    run_gradcam()
