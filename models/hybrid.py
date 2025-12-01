"""
Hybrid CNN-Transformer Architecture Variants (A0, A3, A4, A5)
Ablation study implementations for multi-label chest X-ray classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm


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
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out.expand_as(x)


class SpatialAttention(nn.Module):
    """Spatial attention module"""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        return x * out


# ============================================================================
# A0: Simplified Hybrid - Concatenation Only (No Attention, No Gating)
# ============================================================================

class HybridA0(nn.Module):
    """A0: Simplified hybrid with concatenation fusion only"""
    
    def __init__(self, num_classes=14, pretrained=True, dropout=0.3):
        super(HybridA0, self).__init__()
        self.num_classes = num_classes
        
        # CNN Branch: ResNet-50
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
        
        # Vision Transformer Branch: Swin Tiny
        self.vit = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=pretrained,
            num_classes=0
        )
        vit_channels = self.vit.num_features  # 768 for swin_tiny
        
        # Project ViT features to match CNN spatial size
        self.vit_proj = nn.Sequential(
            nn.Conv2d(vit_channels, cnn_channels, 1),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(inplace=True)
        )
        
        # Simple concatenation + classification
        total_channels = cnn_channels * 2  # CNN + ViT
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(total_channels, num_classes)
        )
    
    def forward_cnn(self, x):
        """Extract CNN features"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # (B, 2048, 16, 16)
        return x
    
    def forward_vit(self, x):
        """Extract ViT features"""
        x = self.vit.forward_features(x)
        
        # Handle different ViT output formats
        if len(x.shape) == 3:  # (B, N, C)
            B, N, C = x.shape
            H = W = int(N ** 0.5)
            x = x.transpose(1, 2).reshape(B, C, H, W)
        
        # Project and resize to match CNN features
        x = self.vit_proj(x)
        if x.size(2) != 16 or x.size(3) != 16:
            x = F.interpolate(x, size=(16, 16), mode='bilinear', align_corners=False)
        
        return x
    
    def forward(self, x):
        cnn_feat = self.forward_cnn(x)
        vit_feat = self.forward_vit(x)
        
        # Simple concatenation
        combined = torch.cat([cnn_feat, vit_feat], dim=1)
        logits = self.classifier(combined)
        return logits


# ============================================================================
# A3: + Channel and Spatial Attention
# ============================================================================

class HybridA3(nn.Module):
    """A3: Concatenation + Channel + Spatial Attention"""
    
    def __init__(self, num_classes=14, pretrained=True, dropout=0.3):
        super(HybridA3, self).__init__()
        self.num_classes = num_classes
        
        # CNN Branch
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
        
        # ViT Branch
        self.vit = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=pretrained,
            num_classes=0
        )
        vit_channels = self.vit.num_features
        
        self.vit_proj = nn.Sequential(
            nn.Conv2d(vit_channels, cnn_channels, 1),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(inplace=True)
        )
        
        # Attention modules for CNN features
        self.cnn_channel_attn = ChannelAttention(cnn_channels)
        self.cnn_spatial_attn = SpatialAttention()
        
        # Attention modules for ViT features
        self.vit_channel_attn = ChannelAttention(cnn_channels)
        self.vit_spatial_attn = SpatialAttention()
        
        # Classifier
        total_channels = cnn_channels * 2
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(total_channels, num_classes)
        )
    
    def forward_cnn(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    
    def forward_vit(self, x):
        x = self.vit.forward_features(x)
        if len(x.shape) == 3:
            B, N, C = x.shape
            H = W = int(N ** 0.5)
            x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.vit_proj(x)
        if x.size(2) != 16 or x.size(3) != 16:
            x = F.interpolate(x, size=(16, 16), mode='bilinear', align_corners=False)
        return x
    
    def forward(self, x):
        cnn_feat = self.forward_cnn(x)
        vit_feat = self.forward_vit(x)
        
        # Apply attention to both branches
        cnn_feat = self.cnn_channel_attn(cnn_feat)
        cnn_feat = self.cnn_spatial_attn(cnn_feat)
        
        vit_feat = self.vit_channel_attn(vit_feat)
        vit_feat = self.vit_spatial_attn(vit_feat)
        
        # Concatenate
        combined = torch.cat([cnn_feat, vit_feat], dim=1)
        logits = self.classifier(combined)
        return logits


# ============================================================================
# A4: + Shared Adaptive Gate
# ============================================================================

class HybridA4(nn.Module):
    """A4: Attention + Shared Adaptive Gate"""
    
    def __init__(self, num_classes=14, pretrained=True, dropout=0.3):
        super(HybridA4, self).__init__()
        self.num_classes = num_classes
        
        # CNN Branch
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
        
        # ViT Branch
        self.vit = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=pretrained,
            num_classes=0
        )
        vit_channels = self.vit.num_features
        
        self.vit_proj = nn.Sequential(
            nn.Conv2d(vit_channels, cnn_channels, 1),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(inplace=True)
        )
        
        # Attention modules
        self.cnn_channel_attn = ChannelAttention(cnn_channels)
        self.cnn_spatial_attn = SpatialAttention()
        self.vit_channel_attn = ChannelAttention(cnn_channels)
        self.vit_spatial_attn = SpatialAttention()
        
        # Shared adaptive gate (learns single alpha for all diseases)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(cnn_channels * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(cnn_channels, num_classes)
        )
    
    def forward_cnn(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    
    def forward_vit(self, x):
        x = self.vit.forward_features(x)
        if len(x.shape) == 3:
            B, N, C = x.shape
            H = W = int(N ** 0.5)
            x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.vit_proj(x)
        if x.size(2) != 16 or x.size(3) != 16:
            x = F.interpolate(x, size=(16, 16), mode='bilinear', align_corners=False)
        return x
    
    def forward(self, x):
        cnn_feat = self.forward_cnn(x)
        vit_feat = self.forward_vit(x)
        
        # Apply attention
        cnn_feat = self.cnn_channel_attn(cnn_feat)
        cnn_feat = self.cnn_spatial_attn(cnn_feat)
        vit_feat = self.vit_channel_attn(vit_feat)
        vit_feat = self.vit_spatial_attn(vit_feat)
        
        # Compute shared gate value
        combined = torch.cat([cnn_feat, vit_feat], dim=1)
        alpha = self.gate(combined).view(-1, 1, 1, 1)
        
        # Adaptive fusion: alpha * CNN + (1-alpha) * ViT
        fused = alpha * cnn_feat + (1 - alpha) * vit_feat
        
        logits = self.classifier(fused)
        return logits


# ============================================================================
# A5: + Per-Class Adaptive Gates (Full Model)
# ============================================================================

class HybridA5(nn.Module):
    """A5: Attention + Per-Class Adaptive Gates (Full Hybrid Model)"""
    
    def __init__(self, num_classes=14, pretrained=True, dropout=0.3):
        super(HybridA5, self).__init__()
        self.num_classes = num_classes
        
        # CNN Branch
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
        
        # ViT Branch
        self.vit = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=pretrained,
            num_classes=0
        )
        vit_channels = self.vit.num_features
        
        self.vit_proj = nn.Sequential(
            nn.Conv2d(vit_channels, cnn_channels, 1),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(inplace=True)
        )
        
        # Attention modules
        self.cnn_channel_attn = ChannelAttention(cnn_channels)
        self.cnn_spatial_attn = SpatialAttention()
        self.vit_channel_attn = ChannelAttention(cnn_channels)
        self.vit_spatial_attn = SpatialAttention()
        
        # Per-class adaptive gates (learns alpha_i for each disease)
        self.disease_gates = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(cnn_channels * 2, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            ) for _ in range(num_classes)
        ])
        
        # Per-class classifiers
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(cnn_channels, 1)
            ) for _ in range(num_classes)
        ])
    
    def forward_cnn(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    
    def forward_vit(self, x):
        x = self.vit.forward_features(x)
        if len(x.shape) == 3:
            B, N, C = x.shape
            H = W = int(N ** 0.5)
            x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.vit_proj(x)
        if x.size(2) != 16 or x.size(3) != 16:
            x = F.interpolate(x, size=(16, 16), mode='bilinear', align_corners=False)
        return x
    
    def forward(self, x, return_gates=False):
        """
        Args:
            x: Input images
            return_gates: If True, return gate values for analysis
        """
        cnn_feat = self.forward_cnn(x)
        vit_feat = self.forward_vit(x)
        
        # Apply attention
        cnn_feat = self.cnn_channel_attn(cnn_feat)
        cnn_feat = self.cnn_spatial_attn(cnn_feat)
        vit_feat = self.vit_channel_attn(vit_feat)
        vit_feat = self.vit_spatial_attn(vit_feat)
        
        # Concatenate for gate computation
        combined = torch.cat([cnn_feat, vit_feat], dim=1)
        
        # Per-disease fusion and classification
        logits = []
        gate_values = []
        
        for i in range(self.num_classes):
            # Compute disease-specific gate
            alpha = self.disease_gates[i](combined).view(-1, 1, 1, 1)
            gate_values.append(alpha.squeeze().detach().cpu())
            
            # Adaptive fusion: alpha_i * CNN + (1-alpha_i) * ViT
            fused = alpha * cnn_feat + (1 - alpha) * vit_feat
            
            # Disease-specific classification
            logit = self.classifiers[i](fused)
            logits.append(logit)
        
        logits = torch.cat(logits, dim=1)
        
        if return_gates:
            return logits, gate_values
        return logits


# ============================================================================
# Factory Function
# ============================================================================

def get_hybrid_model(variant='A0', num_classes=14, pretrained=True, dropout=0.3):
    """
    Factory function to get hybrid model variants
    
    Args:
        variant: 'A0', 'A3', 'A4', 'A5' or 'a0', 'a3', 'a4', 'a5' (case-insensitive)
        num_classes: Number of disease classes
        pretrained: Use pretrained weights
        dropout: Dropout rate
    
    Returns:
        Model instance
    """
    # Handle both uppercase and lowercase, and with/without 'hybrid_' prefix
    variant = str(variant).upper().replace('HYBRID_', '')
    
    if variant == 'A0':
        return HybridA0(num_classes, pretrained, dropout)
    elif variant == 'A3':
        return HybridA3(num_classes, pretrained, dropout)
    elif variant == 'A4':
        return HybridA4(num_classes, pretrained, dropout)
    elif variant == 'A5':
        return HybridA5(num_classes, pretrained, dropout)
    else:
        raise ValueError(f"Unknown variant: {variant}. Choose from 'A0', 'A3', 'A4', 'A5'")


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test all variants
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(2, 3, 512, 512).to(device)
    
    for variant in ['A0', 'A3', 'A4', 'A5']:
        print(f"\nTesting {variant}:")
        model = get_hybrid_model(variant, num_classes=14)
        model = model.to(device)
        
        if variant == 'A5':
            out, gates = model(x, return_gates=True)
            print(f"  Gate values shape: {len(gates)} diseases")
        else:
            out = model(x)
        
        print(f"  Input: {x.shape}")
        print(f"  Output: {out.shape}")
        print(f"  Parameters: {count_parameters(model):,}")