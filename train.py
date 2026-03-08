"""
Main Training Script for GTSRB
Run this script to train the traffic sign recognition model
"""

import torch
import yaml
import argparse
import os
import random
import numpy as np

from src.data import get_data_loaders
from src.models import create_model
from src.training import Trainer


def set_seed(seed: int):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    """Main training function"""
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    set_seed(config['seed'])
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*80}\n")
    
    # Create data loaders
    print("Loading dataset...")
    train_loader, val_loader, test_loader = get_data_loaders(config)
    print(f"✅ Dataset loaded successfully")
    print(f"   Training samples: {len(train_loader.dataset)}")
    print(f"   Validation samples: {len(val_loader.dataset)}")
    print(f"   Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(config, device)
    print(f"✅ Model created successfully")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Train model
    trainer.train()
    
    print("\n✅ Training completed! Check results/ directory for training curves.")
    print(f"✅ Best model saved to: {os.path.join(config['checkpoint']['save_dir'], 'best_model.pth')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GTSRB traffic sign classifier")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    main(args)
