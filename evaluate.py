"""
Model Evaluation Script
Evaluate trained model on test set and generate performance metrics
"""

import torch
import yaml
import argparse
import os
import time
import numpy as np
from tqdm import tqdm

from src.data import get_data_loaders
from src.models import create_model
from src.training.metrics import MetricsTracker, plot_confusion_matrix
from src.utils import get_all_class_names


@torch.no_grad()
def evaluate_model(model, test_loader, device, num_classes=43):
    """
    Evaluate model on test set
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run on
        num_classes: Number of classes
    
    Returns:
        Dictionary of metrics and confusion matrix
    """
    model.eval()
    metrics_tracker = MetricsTracker(num_classes=num_classes)
    
    # Track inference time
    inference_times = []
    
    print("\nEvaluating model on test set...")
    for images, labels in tqdm(test_loader, desc='Testing'):
        images, labels = images.to(device), labels.to(device)
        
        # Measure inference time
        start_time = time.time()
        outputs = model(images)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        inference_time = (time.time() - start_time) / len(images)
        inference_times.append(inference_time)
        
        # Get predictions
        _, preds = torch.max(outputs, 1)
        
        # Update metrics (use dummy loss of 0)
        metrics_tracker.update(preds, labels, 0.0)
    
    # Compute metrics
    metrics = metrics_tracker.compute()
    confusion_mat = metrics_tracker.get_confusion_matrix()
    per_class_acc = metrics_tracker.get_per_class_accuracy()
    
    # Average inference time
    avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms
    
    return metrics, confusion_mat, per_class_acc, avg_inference_time


def main(args):
    """Main evaluation function"""
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load data
    print("Loading test dataset...")
    _, _, test_loader = get_data_loaders(config)
    print(f"✅ Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(config, device)
    
    # Load checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = os.path.join(config['checkpoint']['save_dir'], 'best_model.pth')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Model loaded (trained for {checkpoint['epoch']} epochs)")
    
    # Evaluate
    metrics, confusion_mat, per_class_acc, avg_inference_time = evaluate_model(
        model, test_loader, device, config['data']['num_classes']
    )
    
    # Print results
    print("\n" + "="*80)
    print("Test Set Evaluation Results")
    print("="*80)
    print(f"Test Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision:      {metrics['precision']:.4f}")
    print(f"Recall:         {metrics['recall']:.4f}")
    print(f"F1 Score:       {metrics['f1_score']:.4f}")
    print(f"Num Samples:    {metrics['num_samples']}")
    print(f"\nInference Time: {avg_inference_time:.2f} ms per image")
    print("="*80)
    
    # Find worst performing classes
    sorted_classes = sorted(per_class_acc.items(), key=lambda x: x[1])
    print("\nWorst Performing Classes:")
    class_names = get_all_class_names('en')
    for class_id, acc in sorted_classes[:5]:
        print(f"  Class {class_id:2d} ({class_names[class_id][:40]:40s}): {acc:.4f}")
    
    print("\nBest Performing Classes:")
    for class_id, acc in sorted_classes[-5:]:
        print(f"  Class {class_id:2d} ({class_names[class_id][:40]:40s}): {acc:.4f}")
    
    # Save confusion matrix
    if args.save_cm:
        os.makedirs('results', exist_ok=True)
        cm_path = os.path.join('results', 'confusion_matrix.png')
        plot_confusion_matrix(confusion_mat, cm_path, figsize=(20, 18))
    
    # Benchmark mode
    if args.benchmark:
        print("\n" + "="*80)
        print("Benchmarking Inference Speed")
        print("="*80)
        
        model.eval()
        dummy_input = torch.randn(1, 3, 48, 48).to(device)
        
        # Warmup
        for _ in range(10):
            _ = model(dummy_input)
        
        # Benchmark
        times = []
        for _ in tqdm(range(1000), desc='Benchmarking'):
            start = time.time()
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.time() - start) * 1000)
        
        print(f"\nInference Time Statistics (1000 runs):")
        print(f"  Mean:   {np.mean(times):.2f} ms")
        print(f"  Median: {np.median(times):.2f} ms")
        print(f"  Min:    {np.min(times):.2f} ms")
        print(f"  Max:    {np.max(times):.2f} ms")
        print(f"  Std:    {np.std(times):.2f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GTSRB model")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint (default: best_model.pth)'
    )
    parser.add_argument(
        '--save_cm',
        action='store_true',
        help='Save confusion matrix plot'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run inference speed benchmark'
    )
    
    args = parser.parse_args()
    main(args)
