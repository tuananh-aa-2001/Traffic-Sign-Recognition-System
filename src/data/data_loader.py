"""
GTSRB Dataset Loader
Handles loading images and labels from the GTSRB dataset structure
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Callable
import cv2
from pathlib import Path

from .preprocessing import preprocess_image
from .augmentation import GTSRBAugmentation


class GTSRBDataset(Dataset):
    """
    GTSRB Dataset class for PyTorch
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        config: dict = None,
        transform: Optional[Callable] = None,
        apply_clahe: bool = True
    ):
        """
        Initialize GTSRB dataset
        
        Args:
            root_dir: Root directory containing GTSRB data
            split: 'train', 'val', or 'test'
            config: Configuration dictionary
            transform: Optional transform to apply
            apply_clahe: Whether to apply CLAHE preprocessing
        """
        self.root_dir = root_dir
        self.split = split
        self.config = config or {}
        self.apply_clahe = apply_clahe
        self.image_size = self.config.get('data', {}).get('image_size', 48)
        
        # Set up transforms
        if transform is None:
            is_training = (split == 'train')
            self.transform = GTSRBAugmentation(self.config, is_training=is_training)
        else:
            self.transform = transform
        
        # Load data
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_samples(self) -> list:
        """
        Load image paths and labels
        
        Returns:
            List of (image_path, label) tuples
        """
        samples = []
        
        if self.split in ['train', 'val']:
            # Training data is organized by class folders
            train_dir = os.path.join(self.root_dir, 'GTSRB', 'Final_Training', 'Images')
            
            if not os.path.exists(train_dir):
                raise FileNotFoundError(
                    f"Training directory not found: {train_dir}\n"
                    f"Please run: python src/utils/download_dataset.py"
                )
            
            # Collect all samples from class folders
            all_samples = []
            for class_id in range(43):
                class_dir = os.path.join(train_dir, f'{class_id:05d}')
                if not os.path.exists(class_dir):
                    continue
                
                # Read CSV annotation file
                csv_file = os.path.join(class_dir, f'GT-{class_id:05d}.csv')
                if os.path.exists(csv_file):
                    df = pd.read_csv(csv_file, sep=';')
                    for _, row in df.iterrows():
                        img_path = os.path.join(class_dir, row['Filename'])
                        all_samples.append((img_path, class_id))
            
            # Split into train/val
            np.random.seed(self.config.get('seed', 42))
            indices = np.random.permutation(len(all_samples))
            
            train_split = self.config.get('data', {}).get('train_split', 0.8)
            val_split = self.config.get('data', {}).get('val_split', 0.1)
            
            train_size = int(len(all_samples) * train_split)
            val_size = int(len(all_samples) * val_split)
            
            if self.split == 'train':
                samples = [all_samples[i] for i in indices[:train_size]]
            else:  # val
                samples = [all_samples[i] for i in indices[train_size:train_size + val_size]]
        
        else:  # test
            # Test data has a different structure
            test_dir = os.path.join(self.root_dir, 'GTSRB', 'Final_Test', 'Images')
            csv_file = os.path.join(self.root_dir, 'GTSRB', 'GT-final_test.csv')
            
            if not os.path.exists(test_dir):
                raise FileNotFoundError(
                    f"Test directory not found: {test_dir}\n"
                    f"Please run: python src/utils/download_dataset.py"
                )
            
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file, sep=';')
                for _, row in df.iterrows():
                    img_path = os.path.join(test_dir, row['Filename'])
                    samples.append((img_path, row['ClassId']))
            else:
                # If CSV doesn't exist, just load images without labels
                print("Warning: Test labels not found. Loading images only.")
                for img_file in sorted(os.listdir(test_dir)):
                    if img_file.endswith('.ppm'):
                        img_path = os.path.join(test_dir, img_file)
                        samples.append((img_path, -1))  # -1 for unknown label
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample
        
        Args:
            idx: Sample index
        
        Returns:
            (image_tensor, label) tuple
        """
        img_path, label = self.samples[idx]
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        image = preprocess_image(
            image,
            size=(self.image_size, self.image_size),
            apply_clahe=self.apply_clahe,
            normalize=False  # Augmentation will handle normalization
        )
        
        # Apply transforms
        image = self.transform(image)
        
        return image, label


def get_data_loaders(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders
    
    Args:
        config: Configuration dictionary
    
    Returns:
        (train_loader, val_loader, test_loader) tuple
    """
    root_dir = config['data']['root_dir']
    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']
    pin_memory = config['training']['pin_memory']
    
    # Create datasets
    train_dataset = GTSRBDataset(
        root_dir=root_dir,
        split='train',
        config=config,
        apply_clahe=config['augmentation']['apply_clahe']
    )
    
    val_dataset = GTSRBDataset(
        root_dir=root_dir,
        split='val',
        config=config,
        apply_clahe=config['augmentation']['apply_clahe']
    )
    
    test_dataset = GTSRBDataset(
        root_dir=root_dir,
        split='test',
        config=config,
        apply_clahe=config['augmentation']['apply_clahe']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader
