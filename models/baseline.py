"""
Baseline CNN models: ResNet-50 and DenseNet-121
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50Baseline(nn.Module):
    """ResNet-50 baseline for multi-label classification"""
    
    def __init__(self, num_classes=14, pretrained=True, dropout=0.3):
        """
        Args:
            num_classes: Number of disease classes (14 for ChestX-ray14)
            pretrained: Whether to use ImageNet pretrained weights
            dropout: Dropout rate before final layer
        """
        super(ResNet50Baseline, self).__init__()
        
        # Load pretrained ResNet-50
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # Get number of features from final layer
        num_features = self.resnet.fc.in_features
        
        # Replace final fully connected layer
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes)
        )
        
        self.num_classes = num_classes
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
        
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        logits = self.resnet(x)
        return logits
    
    def get_features(self, x):
        """Extract features before final classification layer"""
        # Forward through all layers except fc
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x


class DenseNet121Baseline(nn.Module):
    """DenseNet-121 baseline (CheXNet architecture)"""
    
    def __init__(self, num_classes=14, pretrained=True, dropout=0.3):
        """
        Args:
            num_classes: Number of disease classes
            pretrained: Whether to use ImageNet pretrained weights
            dropout: Dropout rate before final layer
        """
        super(DenseNet121Baseline, self).__init__()
        
        # Load pretrained DenseNet-121
        self.densenet = models.densenet121(pretrained=pretrained)
        
        # Get number of features from final layer
        num_features = self.densenet.classifier.in_features
        
        # Replace classifier
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes)
        )
        
        self.num_classes = num_classes
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
        
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        logits = self.densenet(x)
        return logits
    
    def get_features(self, x):
        """Extract features before final classification layer"""
        features = self.densenet.features(x)
        out = torch.nn.functional.relu(features, inplace=True)
        out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return out


class MultiScaleResNet(nn.Module):
    """ResNet-50 that extracts multi-scale features"""
    
    def __init__(self, num_classes=14, pretrained=True, dropout=0.3):
        """
        Args:
            num_classes: Number of disease classes
            pretrained: Whether to use ImageNet pretrained weights
            dropout: Dropout rate
        """
        super(MultiScaleResNet, self).__init__()
        
        # Load pretrained ResNet-50
        resnet = models.resnet50(pretrained=pretrained)
        
        # Extract layer components
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # Output: 256 channels, 128x128
        self.layer2 = resnet.layer2  # Output: 512 channels, 64x64
        self.layer3 = resnet.layer3  # Output: 1024 channels, 32x32
        self.layer4 = resnet.layer4  # Output: 2048 channels, 16x16
        
        self.avgpool = resnet.avgpool
        
        # Final classifier
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2048, num_classes)
        
        self.num_classes = num_classes
    
    def forward(self, x, return_features=False):
        """
        Args:
            x: Input tensor of shape (batch_size, 3, 512, 512)
            return_features: If True, return multi-scale features
        
        Returns:
            If return_features=False: logits of shape (batch_size, num_classes)
            If return_features=True: (logits, features_dict)
        """
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Multi-scale features
        feat1 = self.layer1(x)      # 128x128 x 256
        feat2 = self.layer2(feat1)  # 64x64 x 512
        feat3 = self.layer3(feat2)  # 32x32 x 1024
        feat4 = self.layer4(feat3)  # 16x16 x 2048
        
        # Global average pooling and classification
        pooled = self.avgpool(feat4)
        pooled = torch.flatten(pooled, 1)
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        
        if return_features:
            features = {
                'layer1': feat1,
                'layer2': feat2,
                'layer3': feat3,
                'layer4': feat4,
                'pooled': pooled
            }
            return logits, features
        else:
            return logits


def get_model(model_name='resnet50', num_classes=14, pretrained=True, dropout=0.3):
    """
    Factory function to get baseline models
    
    Args:
        model_name: 'resnet50', 'densenet121', or 'resnet50_multiscale'
        num_classes: Number of disease classes
        pretrained: Use ImageNet pretrained weights
        dropout: Dropout rate
    
    Returns:
        Model instance
    """
    model_name = model_name.lower()
    
    if model_name == 'resnet50':
        model = ResNet50Baseline(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout
        )
    elif model_name == 'densenet121':
        model = DenseNet121Baseline(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout
        )
    elif model_name == 'resnet50_multiscale':
        model = MultiScaleResNet(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test ResNet-50
    print("Testing ResNet-50:")
    model = ResNet50Baseline(num_classes=14)
    model = model.to(device)
    x = torch.randn(2, 3, 512, 512).to(device)
    out = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Parameters: {count_parameters(model):,}")
    
    # Test DenseNet-121
    print("\nTesting DenseNet-121:")
    model = DenseNet121Baseline(num_classes=14)
    model = model.to(device)
    out = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Parameters: {count_parameters(model):,}")
    
    # Test MultiScale ResNet
    print("\nTesting MultiScale ResNet-50:")
    model = MultiScaleResNet(num_classes=14)
    model = model.to(device)
    out, features = model(x, return_features=True)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Feature shapes:")
    for name, feat in features.items():
        print(f"    {name}: {feat.shape}")
    print(f"  Parameters: {count_parameters(model):,}")
