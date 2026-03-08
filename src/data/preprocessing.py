"""
Image Preprocessing Utilities for GTSRB
Includes CLAHE histogram equalization and normalization
"""

import cv2
import numpy as np
from typing import Tuple


class CLAHEPreprocessor:
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE) preprocessor
    Improves image quality under varying lighting conditions
    """
    
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        """
        Initialize CLAHE preprocessor
        
        Args:
            clip_limit: Threshold for contrast limiting
            tile_grid_size: Size of grid for histogram equalization
        """
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE to an image
        
        Args:
            image: Input image (RGB or grayscale)
        
        Returns:
            Preprocessed image
        """
        # Convert to LAB color space for better results
        if len(image.shape) == 3 and image.shape[2] == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            # Grayscale image
            return self.clahe.apply(image)


def resize_image(image: np.ndarray, size: Tuple[int, int] = (48, 48)) -> np.ndarray:
    """
    Resize image to target size
    
    Args:
        image: Input image
        size: Target size (width, height)
    
    Returns:
        Resized image
    """
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def normalize_image(image: np.ndarray, mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> np.ndarray:
    """
    Normalize image using ImageNet statistics
    
    Args:
        image: Input image (0-255 range)
        mean: Mean values for each channel
        std: Standard deviation for each channel
    
    Returns:
        Normalized image
    """
    # Convert to float and scale to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Normalize
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    
    return (image - mean) / std


def preprocess_image(image: np.ndarray, 
                     size: Tuple[int, int] = (48, 48),
                     apply_clahe: bool = True,
                     normalize: bool = True) -> np.ndarray:
    """
    Complete preprocessing pipeline for GTSRB images
    
    Args:
        image: Input image (RGB, 0-255 range)
        size: Target size
        apply_clahe: Whether to apply CLAHE
        normalize: Whether to normalize
    
    Returns:
        Preprocessed image
    """
    # Apply CLAHE if requested
    if apply_clahe:
        clahe = CLAHEPreprocessor()
        image = clahe(image)
    
    # Resize
    image = resize_image(image, size)
    
    # Normalize if requested
    if normalize:
        image = normalize_image(image)
    
    return image
