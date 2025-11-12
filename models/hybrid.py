"""
Hybrid CNN-Transformer Architecture with Adaptive Attention Fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm  # For Vision Transformer models


class ChannelAttention(nn.Module):
    """Channel attention module (Squeeze-and-Excitation)"""
    
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Average and max pooling
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        # Combine and apply sigmoid
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out.expand_as(x)


class SpatialAttention(nn.Module):
    """Spatial attention module"""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Average and max pooling across channels
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and convolve
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        
        return x * out


class AdaptiveAttentionFusion(nn.Module):
    """
    Adaptive attention fusion module that combines CNN and Transformer features
    with disease-specific gating
    """
    
    def __init__(self, cnn_channels, vit_channels, num_classes=14):
        super(AdaptiveAttentionFusion, self).__init__()
        
        self.num_classes = num_classes
        
        # Project features to same dimension if different
        if cnn_channels != vit_channels:
            self.cnn_proj = nn.Conv2d(cnn_channels, vit_channels, 1)
        else:
            self.cnn_proj = nn.Identity()
        
        # Channel and spatial attention for CNN features
        self.cnn_channel_attn = ChannelAttention(vit_channels)
        self.cnn_spatial_attn = SpatialAttention()
        
        # Channel and spatial attention for ViT features
        self.vit_channel_attn = ChannelAttention(vit_channels)
        self.vit_spatial_attn = SpatialAttention()
        
        # Disease-specific gating: learns alpha_i for each disease
        self.disease_gates = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(vit_channels * 2, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            ) for _ in range(num_classes)
        ])
    
    def forward(self, cnn_feat, vit_feat):
        """
        Args:
            cnn_feat: CNN features (B, C1, H, W)
            vit_feat: ViT features (B, C2, H, W)
        
        Returns:
            fused_features: List of fused features for each disease class
        """
        batch_size = cnn_feat.size(0)
        
        # Project CNN features if needed
        cnn_feat = self.cnn_proj(cnn_feat)
        
        # Apply attention to both feature types
        cnn_feat_attn = self.cnn_channel_attn(cnn_feat)
        cnn_feat_attn = self.cnn_spatial_attn(cnn_feat_attn)
        
        vit_feat_attn = self.vit_channel_attn(vit_feat)
        vit_feat_attn = self.vit_spatial_attn(vit_feat_attn)
        
        # Concatenate for gate computation
        combined = torch.cat([cnn_feat_attn, vit_feat_attn], dim=1)
        
        # Disease-specific fusion
        fused_features = []
        for gate in self.disease_gates:
            # Compute gate value (alpha) for this disease
            alpha = gate(combined).view(batch_size, 1, 1, 1)
            
            # Weighted combination: alpha * CNN + (1-alpha) * ViT
            fused = alpha * cnn_feat_attn + (1 - alpha) * vit_feat_attn
            fused_features.append(fused)
        
        return fused_features


class HybridCNNTransformer(nn.Module):
    """
    Hybrid CNN-Transformer architecture for multi-label chest X-ray classification
    """
    
    def __init__(self, 
                 num_classes=14,
                 cnn_model='resnet50',
                 vit_model='swin_tiny_patch4_window7_224',
                 pretrained=True,
                 dropout=0.3):
        """
        Args:
            num_classes: Number of disease classes
            cnn_model: CNN backbone ('resnet50', 'densenet121')
            vit_model: Vision Transformer model name from timm
            pretrained: Use pretrained weights
            dropout: Dropout rate
        """
        super(HybridCNNTransformer, self).__init__()
        
        self.num_classes = num_classes
        
        # CNN Backbone (extract multi-scale features)
        if cnn_model == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            self.conv1 = resnet.conv1
            self.bn1 = resnet.bn1
            self.relu = resnet.relu
            self.maxpool = resnet.maxpool
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3
            self.layer4 = resnet.layer4
            cnn_channels = 2048
        else:
            raise NotImplementedError(f"CNN model {cnn_model} not implemented")
        
        # Vision Transformer Branch
        self.vit = timm.create_model(vit_model, pretrained=pretrained, num_classes=0)
        vit_channels = self.vit.num_features
        
        # Projection to match CNN feature dimensions
        self.vit_proj = nn.Sequential(
            nn.Conv2d(vit_channels, cnn_channels, 1),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(inplace=True)
        )
        
        # Adaptive Attention Fusion
        self.fusion = AdaptiveAttentionFusion(
            cnn_channels=cnn_channels,
            vit_channels=cnn_channels,
            num_classes=num_classes
        )
        
        # Disease-specific classification heads
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(cnn_channels, 1)
            ) for _ in range(num_classes)
        ])
    
    def forward_cnn(self, x):
        """Extract CNN features"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # Shape: (B, 2048, 16, 16) for 512x512 input
        
        return x
    
    def forward_vit(self, x):
        """Extract ViT features and reshape to spatial format"""
        # ViT forward
        x = self.vit.forward_features(x)
        
        # Reshape from (B, N, C) to (B, C, H, W)
        # For Swin Transformer, output is already spatial
        if len(x.shape) == 3:  # (B, N, C) format
            B, N, C = x.shape
            H = W = int(N ** 0.5)
            x = x.transpose(1, 2).reshape(B, C, H, W)
        
        # Project to CNN feature dimension
        x = self.vit_proj(x)
        
        # Upsample to match CNN feature size if needed
        # CNN features are 16x16, so we may need to upsample ViT features
        if x.size(2) != 16 or x.size(3) != 16:
            x = F.interpolate(x, size=(16, 16), mode='bilinear', align_corners=False)
        
        return x
    
    def forward(self, x):
        """
        Args:
            x: Input images (B, 3, H, W)
        
        Returns:
            logits: Tensor of shape (B, num_classes)
        """
        # Extract features from both branches
        cnn_feat = self.forward_cnn(x)
        vit_feat = self.forward_vit(x)
        
        # Adaptive fusion for each disease
        fused_features = self.fusion(cnn_feat, vit_feat)
        
        # Disease-specific classification
        logits = []
        for i, classifier in enumerate(self.classifiers):
            logit = classifier(fused_features[i])
            logits.append(logit)
        
        logits = torch.cat(logits, dim=1)  # (B, num_classes)
        
        return logits


class SimplifiedHybrid(nn.Module):
    """Simplified hybrid model with concatenation fusion (ablation baseline)"""
    
    def __init__(self, 
                 num_classes=14,
                 cnn_model='resnet50',
                 vit_model='swin_tiny_patch4_window7_224',
                 pretrained=True,
                 dropout=0.3):
        super(SimplifiedHybrid, self).__init__()
        
        self.num_classes = num_classes
        
        # CNN Backbone
        resnet = models.resnet50(pretrained=pretrained)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool and fc
        cnn_channels = 2048
        
        # Vision Transformer
        self.vit = timm.create_model(vit_model, pretrained=pretrained, num_classes=0)
        vit_channels = self.vit.num_features
        
        # Simple concatenation fusion
        total_channels = cnn_channels + vit_channels
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(total_channels, num_classes)
        )
    
    def forward(self, x):
        # CNN features
        cnn_feat = self.cnn(x)
        
        # ViT features
        vit_feat = self.vit.forward_features(x)
        if len(vit_feat.shape) == 3:
            B, N, C = vit_feat.shape
            H = W = int(N ** 0.5)
            vit_feat = vit_feat.transpose(1, 2).reshape(B, C, H, W)
        
        # Resize ViT features to match CNN
        if vit_feat.size(2) != cnn_feat.size(2):
            vit_feat = F.interpolate(
                vit_feat, 
                size=(cnn_feat.size(2), cnn_feat.size(3)),
                mode='bilinear',
                align_corners=False
            )
        
        # Concatenate
        combined = torch.cat([cnn_feat, vit_feat], dim=1)
        
        # Classify
        logits = self.classifier(combined)
        
        return logits


def get_hybrid_model(model_type='full', num_classes=14, pretrained=True, dropout=0.3):
    """
    Factory function for hybrid models
    
    Args:
        model_type: 'full' (with adaptive attention) or 'simple' (concatenation only)
        num_classes: Number of disease classes
        pretrained: Use pretrained weights
        dropout: Dropout rate
    
    Returns:
        Model instance
    """
    if model_type == 'full':
        return HybridCNNTransformer(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout
        )
    elif model_type == 'simple':
        return SimplifiedHybrid(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == '__main__':
    # Test hybrid model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Testing Full Hybrid Model:")
    model = get_hybrid_model('full', num_classes=14)
    model = model.to(device)
    
    x = torch.randn(2, 3, 512, 512).to(device)
    out = model(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
