"""
Training Loop and Trainer Class
Handles model training, validation, checkpointing, and logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import os
from tqdm import tqdm
from typing import Dict, Tuple
import time

from .metrics import MetricsTracker, EarlyStopping, plot_training_curves


class Trainer:
    """
    Trainer class for GTSRB model training
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config: dict,
        device: torch.device
    ):
        """
        Initialize trainer
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration dictionary
            device: Device to train on
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Training settings
        self.num_epochs = config['training']['num_epochs']
        self.log_interval = config['logging']['log_interval']
        
        # Loss function with label smoothing
        label_smoothing = config['loss'].get('label_smoothing', 0.1)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision training
        self.use_amp = config['mixed_precision']['enabled']
        self.scaler = GradScaler() if self.use_amp else None
        
        # Early stopping
        early_stop_config = config['early_stopping']
        self.early_stopping = EarlyStopping(
            patience=early_stop_config['patience'],
            min_delta=early_stop_config['min_delta'],
            mode='max'
        )
        
        # Checkpointing
        self.checkpoint_dir = config['checkpoint']['save_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # TensorBoard logging
        tensorboard_dir = config['logging']['tensorboard_dir']
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(tensorboard_dir)
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0.0
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer"""
        opt_config = self.config['optimizer']
        lr = self.config['training']['learning_rate']
        weight_decay = self.config['training']['weight_decay']
        
        if opt_config['type'].lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                betas=opt_config['betas'],
                weight_decay=weight_decay
            )
        elif opt_config['type'].lower() == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config['type']}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        sched_config = self.config['scheduler']
        
        if sched_config['type'] == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=sched_config['mode'],
                factor=sched_config['factor'],
                patience=sched_config['patience'],
                min_lr=sched_config['min_lr']
            )
        else:
            return None
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        metrics_tracker = MetricsTracker(num_classes=self.config['data']['num_classes'])
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.num_epochs} [Train]')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            
            # Update metrics
            metrics_tracker.update(preds, labels, loss.item())
            
            # Update progress bar
            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Compute epoch metrics
        metrics = metrics_tracker.compute()
        
        return metrics
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate the model
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        metrics_tracker = MetricsTracker(num_classes=self.config['data']['num_classes'])
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch}/{self.num_epochs} [Val]')
        
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            
            # Update metrics
            metrics_tracker.update(preds, labels, loss.item())
        
        # Compute epoch metrics
        metrics = metrics_tracker.compute()
        
        return metrics
    
    def save_checkpoint(self, epoch: int, val_acc: float, is_best: bool = False):
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch
            val_acc: Validation accuracy
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'config': self.config
        }
        
        if is_best:
            path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, path)
            print(f"✅ Best model saved with validation accuracy: {val_acc:.4f}")
    
    def train(self):
        """
        Main training loop
        """
        print("\n" + "="*80)
        print("Starting Training")
        print("="*80)
        
        start_time = time.time()
        
        for epoch in range(1, self.num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Store metrics
            self.train_losses.append(train_metrics['loss'])
            self.val_losses.append(val_metrics['loss'])
            self.train_accs.append(train_metrics['accuracy'])
            self.val_accs.append(val_metrics['accuracy'])
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            self.writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{self.num_epochs}:")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f} | Val Acc:   {val_metrics['accuracy']:.4f}")
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step(val_metrics['accuracy'])
                print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.save_checkpoint(epoch, val_metrics['accuracy'], is_best=True)
            
            # Early stopping
            if self.early_stopping(val_metrics['accuracy']):
                print(f"\n⚠️  Early stopping triggered after {epoch} epochs")
                break
        
        # Training complete
        elapsed_time = time.time() - start_time
        print("\n" + "="*80)
        print("Training Complete!")
        print("="*80)
        print(f"Total time: {elapsed_time/60:.2f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        
        # Plot training curves
        curves_path = os.path.join('results', 'training_curves.png')
        os.makedirs('results', exist_ok=True)
        plot_training_curves(
            self.train_losses,
            self.val_losses,
            self.train_accs,
            self.val_accs,
            curves_path
        )
        
        self.writer.close()
