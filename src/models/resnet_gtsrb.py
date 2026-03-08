"""
Custom ResNet-based Model for GTSRB
Optimized for 48x48 input images and 43-class classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from typing import Optional


class ResNetGTSRB(nn.Module):
    """
    Custom ResNet-18 based model optimized for GTSRB
    Modified for small 48x48 images with 43 output classes
    """
    
    def __init__(
        self,
        num_classes: int = 43,
        pretrained: bool = False,
        dropout: float = 0.5
    ):
        """
        Initialize ResNet model for GTSRB
        
        Args:
            num_classes: Number of output classes (43 for GTSRB)
            pretrained: Whether to use ImageNet pretrained weights
            dropout: Dropout probability before final classifier
        """
        super(ResNetGTSRB, self).__init__()
        
        # Load base ResNet-18
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
            base_model = resnet18(weights=weights)
        else:
            base_model = resnet18(weights=None)
        
        # Modify first conv layer for better performance on small images
        # Original: 7x7 kernel with stride 2
        # Modified: 3x3 kernel with stride 1 (better for 48x48 images)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        
        # Remove maxpool to preserve spatial information for small images
        # self.maxpool = base_model.maxpool  # Skip this
        
        # ResNet layers
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout)
        
        # Final classifier
        self.fc = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for new layers"""
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.constant_(self.fc.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, 3, 48, 48)
        
        Returns:
            Output logits (batch_size, num_classes)
        """
        # Initial conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # Skip maxpool for small images
        
        # ResNet blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Dropout and classifier
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LightweightCNN(nn.Module):
    """
    Lightweight CNN alternative for faster inference
    Custom architecture optimized for speed
    """
    
    def __init__(self, num_classes: int = 43, dropout: float = 0.5):
        """
        Initialize lightweight CNN
        
        Args:
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super(LightweightCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 48x48 -> 24x24
        
        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 24x24 -> 12x12
        
        # Conv block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 12x12 -> 6x6
        
        # Conv block 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # 6x6 -> 3x3
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Fully connected
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config: dict, device: torch.device) -> nn.Module:
    """
    Create model based on configuration
    
    Args:
        config: Configuration dictionary
        device: Device to place model on
    
    Returns:
        Model instance
    """
    model_config = config['model']
    architecture = model_config.get('architecture', 'resnet18')
    num_classes = config['data']['num_classes']
    dropout = model_config.get('dropout', 0.5)
    pretrained = model_config.get('pretrained', False)
    
    if architecture == 'resnet18':
        model = ResNetGTSRB(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout
        )
    elif architecture == 'lightweight':
        model = LightweightCNN(
            num_classes=num_classes,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    model = model.to(device)
    
    print(f"\nModel: {architecture}")
    print(f"Total parameters: {model.get_num_parameters():,}")
    print(f"Device: {device}")
    
    return model
