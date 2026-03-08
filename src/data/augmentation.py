"""
Data Augmentation for GTSRB
Simulates real-world conditions: camera vibration, varying angles, lighting
"""

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
import numpy as np
from typing import Tuple


class RandomRotation:
    """Random rotation within specified degrees"""
    
    def __init__(self, degrees: float = 15):
        self.degrees = degrees
    
    def __call__(self, img):
        angle = random.uniform(-self.degrees, self.degrees)
        return TF.rotate(img, angle)


class RandomTranslation:
    """Random translation (shift) of the image"""
    
    def __init__(self, translate_percent: float = 0.1):
        self.translate_percent = translate_percent
    
    def __call__(self, img):
        width, height = img.size
        max_dx = int(width * self.translate_percent)
        max_dy = int(height * self.translate_percent)
        
        dx = random.randint(-max_dx, max_dx)
        dy = random.randint(-max_dy, max_dy)
        
        return TF.affine(img, angle=0, translate=(dx, dy), scale=1.0, shear=0)


class RandomScale:
    """Random scaling of the image"""
    
    def __init__(self, scale_range: Tuple[float, float] = (0.9, 1.1)):
        self.scale_range = scale_range
    
    def __call__(self, img):
        scale = random.uniform(*self.scale_range)
        return TF.affine(img, angle=0, translate=(0, 0), scale=scale, shear=0)


class GaussianNoise:
    """Add Gaussian noise to simulate sensor noise"""
    
    def __init__(self, std: float = 0.01):
        self.std = std
    
    def __call__(self, tensor):
        if random.random() > 0.5:
            noise = torch.randn_like(tensor) * self.std
            return tensor + noise
        return tensor


def get_train_transforms(config: dict) -> T.Compose:
    """
    Get training data augmentation transforms
    
    Args:
        config: Configuration dictionary with augmentation parameters
    
    Returns:
        Composed transforms
    """
    aug_config = config.get('augmentation', {})
    
    transforms = [
        T.ToPILImage(),
        RandomRotation(degrees=aug_config.get('rotation_degrees', 15)),
        RandomTranslation(translate_percent=aug_config.get('translation_percent', 0.1)),
        RandomScale(scale_range=aug_config.get('scale_range', [0.9, 1.1])),
        T.ColorJitter(
            brightness=aug_config.get('brightness_factor', 0.2),
            contrast=aug_config.get('contrast_factor', 0.2),
            saturation=0.2,
            hue=0.1
        ),
        T.ToTensor(),
    ]
    
    # Add Gaussian noise after converting to tensor
    if aug_config.get('gaussian_noise_std', 0) > 0:
        transforms.append(GaussianNoise(std=aug_config['gaussian_noise_std']))
    
    # Normalize (ImageNet statistics)
    transforms.append(
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )
    
    return T.Compose(transforms)


def get_test_transforms() -> T.Compose:
    """
    Get test/validation transforms (no augmentation)
    
    Returns:
        Composed transforms
    """
    return T.Compose([
        T.ToPILImage(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class GTSRBAugmentation:
    """
    Complete augmentation pipeline for GTSRB
    """
    
    def __init__(self, config: dict, is_training: bool = True):
        """
        Initialize augmentation pipeline
        
        Args:
            config: Configuration dictionary
            is_training: Whether this is for training (applies augmentation)
        """
        self.is_training = is_training
        
        if is_training:
            self.transform = get_train_transforms(config)
        else:
            self.transform = get_test_transforms()
    
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """
        Apply transforms to image
        
        Args:
            image: Input image (numpy array, RGB, 0-255 or 0-1 range)
        
        Returns:
            Transformed tensor
        """
        # Ensure image is in 0-255 range for PIL conversion
        if image.dtype == np.float32 or image.dtype == np.float64:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
        
        return self.transform(image)
