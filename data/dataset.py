"""
PyTorch Dataset class for NIH ChestX-ray14
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
import torchvision.transforms as transforms


class ChestXrayDataset(Dataset):
    """NIH ChestX-ray14 Dataset"""
    
    def __init__(self, 
                 csv_file, 
                 image_dir, 
                 transform=None,
                 disease_classes=None):
        """
        Args:
            csv_file: Path to CSV file with image names and labels
            image_dir: Directory with all the images
            transform: Optional transform to be applied on images
            disease_classes: List of disease class names
        """
        self.df = pd.read_csv(csv_file)
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        if disease_classes is None:
            self.disease_classes = [
                'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
                'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
                'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
                'Pleural_Thickening', 'Hernia'
            ]
        else:
            self.disease_classes = disease_classes
        
        self.num_classes = len(self.disease_classes)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Tensor of shape (3, H, W)
            labels: Tensor of shape (num_classes,) with binary labels
            image_name: String with image filename
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image name and path
        img_name = self.df.iloc[idx]['Image Index']
        
        # Images are organized in folders like images_01/images/00000001_000.png
        # Find which folder contains the image
        img_path = None
        
        # Check all images_XX folders
        for folder in sorted(self.image_dir.glob('images_*')):
            if not folder.is_dir():
                continue
            potential_path = folder / 'images' / img_name
            if potential_path.exists():
                img_path = potential_path
                break
        
        # Fallback: try direct path
        if img_path is None:
            img_path = self.image_dir / img_name
        
        # Final check
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_name}\nSearched in: {self.image_dir}")
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Get labels
        labels = []
        for disease in self.disease_classes:
            labels.append(self.df.iloc[idx][disease])
        labels = torch.FloatTensor(labels)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, labels, img_name


def get_transforms(image_size=512, augment=True):
    """
    Get train and validation transforms
    
    Args:
        image_size: Size to resize images to
        augment: Whether to apply data augmentation (for training)
    
    Returns:
        transforms.Compose object
    """
    if augment:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        # Validation/Test transforms (no augmentation)
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    return transform


def get_dataloaders(data_dir, 
                   image_dir,
                   batch_size=16,
                   num_workers=4,
                   image_size=512):
    """
    Create train, validation, and test dataloaders
    
    Args:
        data_dir: Directory containing train.csv, val.csv, test.csv
        image_dir: Directory containing images
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        image_size: Size to resize images to
    
    Returns:
        train_loader, val_loader, test_loader
    """
    data_dir = Path(data_dir)
    
    # Create datasets
    train_dataset = ChestXrayDataset(
        csv_file=data_dir / 'train.csv',
        image_dir=image_dir,
        transform=get_transforms(image_size=image_size, augment=True)
    )
    
    val_dataset = ChestXrayDataset(
        csv_file=data_dir / 'val.csv',
        image_dir=image_dir,
        transform=get_transforms(image_size=image_size, augment=False)
    )
    
    test_dataset = ChestXrayDataset(
        csv_file=data_dir / 'test.csv',
        image_dir=image_dir,
        transform=get_transforms(image_size=image_size, augment=False)
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


# CLAHE augmentation (optional advanced preprocessing)
class CLAHETransform:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
    
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        import cv2
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size
        )
    
    def __call__(self, img):
        """
        Args:
            img: PIL Image
        Returns:
            PIL Image with CLAHE applied
        """
        import cv2
        import numpy as np
        
        # Convert PIL to numpy
        img_np = np.array(img)
        
        # Apply CLAHE to each channel if RGB
        if len(img_np.shape) == 3:
            img_np[:, :, 0] = self.clahe.apply(img_np[:, :, 0])
            img_np[:, :, 1] = self.clahe.apply(img_np[:, :, 1])
            img_np[:, :, 2] = self.clahe.apply(img_np[:, :, 2])
        else:
            img_np = self.clahe.apply(img_np)
        
        # Convert back to PIL
        return Image.fromarray(img_np)


def get_transforms_with_clahe(image_size=512, augment=True):
    """Get transforms including CLAHE"""
    if augment:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            CLAHETransform(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            CLAHETransform(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    return transform