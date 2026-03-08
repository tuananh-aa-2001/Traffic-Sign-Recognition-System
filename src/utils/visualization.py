"""
Visualization Utilities
Plot sample predictions, training curves, and confusion matrices
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import cv2
from typing import List, Tuple
import os

from src.utils import get_class_name


def denormalize_image(img_tensor: torch.Tensor) -> np.ndarray:
    """
    Denormalize image tensor for visualization
    
    Args:
        img_tensor: Normalized image tensor (C, H, W)
    
    Returns:
        Denormalized image array (H, W, C) in 0-255 range
    """
    # ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Convert to numpy and denormalize
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * std + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    
    return img


def visualize_predictions(
    model,
    data_loader,
    device,
    num_samples: int = 16,
    save_path: str = 'results/sample_predictions.png'
):
    """
    Visualize model predictions on sample images
    
    Args:
        model: Trained model
        data_loader: Data loader
        device: Device
        num_samples: Number of samples to visualize
        save_path: Path to save figure
    """
    model.eval()
    
    images_list = []
    labels_list = []
    preds_list = []
    probs_list = []
    
    # Collect samples
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            max_probs, preds = torch.max(probs, 1)
            
            images_list.extend(images.cpu())
            labels_list.extend(labels.cpu().numpy())
            preds_list.extend(preds.cpu().numpy())
            probs_list.extend(max_probs.cpu().numpy())
            
            if len(images_list) >= num_samples:
                break
    
    # Plot
    rows = int(np.sqrt(num_samples))
    cols = int(np.ceil(num_samples / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.5))
    axes = axes.flatten()
    
    for idx in range(num_samples):
        ax = axes[idx]
        
        # Denormalize and display image
        img = denormalize_image(images_list[idx])
        ax.imshow(img)
        
        # Get labels
        true_label = labels_list[idx]
        pred_label = preds_list[idx]
        confidence = probs_list[idx]
        
        # Get class names
        true_name = get_class_name(true_label, 'en')
        pred_name = get_class_name(pred_label, 'en')
        
        # Color code: green if correct, red if wrong
        color = 'green' if true_label == pred_label else 'red'
        
        # Title
        title = f"True: {true_name[:25]}\n"
        title += f"Pred: {pred_name[:25]}\n"
        title += f"Conf: {confidence:.2f}"
        
        ax.set_title(title, fontsize=9, color=color, weight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Sample predictions saved to: {save_path}")


def visualize_misclassified(
    model,
    data_loader,
    device,
    num_samples: int = 16,
    save_path: str = 'results/misclassified_samples.png'
):
    """
    Visualize misclassified examples
    
    Args:
        model: Trained model
        data_loader: Data loader
        device: Device
        num_samples: Number of samples to visualize
        save_path: Path to save figure
    """
    model.eval()
    
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []
    misclassified_probs = []
    
    # Collect misclassified samples
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            max_probs, preds = torch.max(probs, 1)
            
            # Find misclassified
            mask = preds != labels
            
            if mask.any():
                misclassified_images.extend(images[mask].cpu())
                misclassified_labels.extend(labels[mask].cpu().numpy())
                misclassified_preds.extend(preds[mask].cpu().numpy())
                misclassified_probs.extend(max_probs[mask].cpu().numpy())
            
            if len(misclassified_images) >= num_samples:
                break
    
    if len(misclassified_images) == 0:
        print("No misclassified samples found!")
        return
    
    # Plot
    num_to_plot = min(num_samples, len(misclassified_images))
    rows = int(np.sqrt(num_to_plot))
    cols = int(np.ceil(num_to_plot / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.5))
    if num_to_plot == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx in range(num_to_plot):
        ax = axes[idx]
        
        # Denormalize and display image
        img = denormalize_image(misclassified_images[idx])
        ax.imshow(img)
        
        # Get labels
        true_label = misclassified_labels[idx]
        pred_label = misclassified_preds[idx]
        confidence = misclassified_probs[idx]
        
        # Get class names
        true_name = get_class_name(true_label, 'en')
        pred_name = get_class_name(pred_label, 'en')
        
        # Title
        title = f"True: {true_name[:25]}\n"
        title += f"Pred: {pred_name[:25]}\n"
        title += f"Conf: {confidence:.2f}"
        
        ax.set_title(title, fontsize=9, color='red', weight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Misclassified samples saved to: {save_path}")
