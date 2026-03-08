"""
Training Metrics and Evaluation
Tracks accuracy, loss, confusion matrix, and per-class performance
"""

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class MetricsTracker:
    """Track and compute training metrics"""
    
    def __init__(self, num_classes: int = 43):
        """
        Initialize metrics tracker
        
        Args:
            num_classes: Number of classes
        """
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.all_preds = []
        self.all_labels = []
        self.running_loss = 0.0
        self.num_samples = 0
    
    def update(self, preds: torch.Tensor, labels: torch.Tensor, loss: float):
        """
        Update metrics with batch results
        
        Args:
            preds: Predicted class indices (batch_size,)
            labels: True labels (batch_size,)
            loss: Batch loss value
        """
        self.all_preds.extend(preds.cpu().numpy().tolist())
        self.all_labels.extend(labels.cpu().numpy().tolist())
        self.running_loss += loss * len(labels)
        self.num_samples += len(labels)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics
        
        Returns:
            Dictionary of metric names and values
        """
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)
        
        # Overall accuracy
        accuracy = accuracy_score(labels, preds)
        
        # Average loss
        avg_loss = self.running_loss / max(self.num_samples, 1)
        
        # Top-3 and Top-5 accuracy (requires logits, not implemented here)
        # For now, just return top-1 accuracy
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, preds, average='macro', zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'loss': avg_loss,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'num_samples': self.num_samples
        }
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """
        Get confusion matrix
        
        Returns:
            Confusion matrix (num_classes, num_classes)
        """
        return confusion_matrix(
            self.all_labels,
            self.all_preds,
            labels=list(range(self.num_classes))
        )
    
    def get_per_class_accuracy(self) -> Dict[int, float]:
        """
        Get per-class accuracy
        
        Returns:
            Dictionary mapping class_id to accuracy
        """
        cm = self.get_confusion_matrix()
        per_class_acc = {}
        
        for i in range(self.num_classes):
            if cm[i].sum() > 0:
                per_class_acc[i] = cm[i, i] / cm[i].sum()
            else:
                per_class_acc[i] = 0.0
        
        return per_class_acc
    
    def print_classification_report(self, class_names: List[str] = None):
        """
        Print detailed classification report
        
        Args:
            class_names: Optional list of class names
        """
        if class_names is None:
            class_names = [f"Class {i}" for i in range(self.num_classes)]
        
        report = classification_report(
            self.all_labels,
            self.all_preds,
            target_names=class_names,
            zero_division=0
        )
        
        print("\n" + "="*80)
        print("Classification Report")
        print("="*80)
        print(report)


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: str,
    class_names: List[str] = None,
    figsize: Tuple[int, int] = (20, 18)
):
    """
    Plot and save confusion matrix heatmap
    
    Args:
        cm: Confusion matrix
        save_path: Path to save figure
        class_names: Optional class names
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    
    # Plot heatmap
    sns.heatmap(
        cm_normalized,
        annot=False,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names if class_names else range(cm.shape[0]),
        yticklabels=class_names if class_names else range(cm.shape[0]),
        cbar_kws={'label': 'Accuracy'}
    )
    
    plt.title('Confusion Matrix (Normalized)', fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to: {save_path}")


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: str
):
    """
    Plot training and validation curves
    
    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        train_accs: Training accuracies per epoch
        val_accs: Validation accuracies per epoch
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to: {save_path}")


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'max'):
        """
        Initialize early stopping
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics to maximize, 'min' for metrics to minimize
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if should stop training
        
        Args:
            score: Current metric value
        
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > (self.best_score + self.min_delta)
        else:
            improved = score < (self.best_score - self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False
