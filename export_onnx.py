"""
ONNX Export Script
Export trained PyTorch model to ONNX format for deployment
"""

import torch
import onnx
import onnxruntime as ort
import numpy as np
import yaml
import argparse
import os

from src.models import create_model


def export_to_onnx(model, onnx_path, input_shape=(1, 3, 48, 48)):
    """
    Export PyTorch model to ONNX format
    
    Args:
        model: PyTorch model
        onnx_path: Path to save ONNX model
        input_shape: Input tensor shape
    """
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape)
    
    # Export to ONNX
    print(f"\nExporting model to ONNX format...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✅ Model exported to: {onnx_path}")


def verify_onnx_model(pytorch_model, onnx_path, device):
    """
    Verify that ONNX model produces same outputs as PyTorch model
    
    Args:
        pytorch_model: Original PyTorch model
        onnx_path: Path to ONNX model
        device: Device for PyTorch model
    
    Returns:
        True if outputs match, False otherwise
    """
    print("\nVerifying ONNX model...")
    
    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX model is valid")
    
    # Create ONNX Runtime session
    ort_session = ort.InferenceSession(onnx_path)
    
    # Create test input
    test_input = torch.randn(1, 3, 48, 48)
    
    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input.to(device)).cpu().numpy()
    
    # ONNX inference
    ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
    onnx_output = ort_session.run(None, ort_inputs)[0]
    
    # Compare outputs
    max_diff = np.abs(pytorch_output - onnx_output).max()
    mean_diff = np.abs(pytorch_output - onnx_output).mean()
    
    print(f"\nOutput Comparison:")
    print(f"  Max difference:  {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")
    
    if max_diff < 1e-5:
        print("✅ ONNX model outputs match PyTorch model!")
        return True
    else:
        print("⚠️  Warning: ONNX outputs differ from PyTorch")
        return False


def benchmark_onnx(onnx_path, num_runs=1000):
    """
    Benchmark ONNX model inference speed
    
    Args:
        onnx_path: Path to ONNX model
        num_runs: Number of inference runs
    """
    import time
    
    print(f"\nBenchmarking ONNX model ({num_runs} runs)...")
    
    # Create session
    ort_session = ort.InferenceSession(onnx_path)
    
    # Create test input
    test_input = np.random.randn(1, 3, 48, 48).astype(np.float32)
    ort_inputs = {ort_session.get_inputs()[0].name: test_input}
    
    # Warmup
    for _ in range(10):
        _ = ort_session.run(None, ort_inputs)
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.time()
        _ = ort_session.run(None, ort_inputs)
        times.append((time.time() - start) * 1000)
    
    times = np.array(times)
    
    print(f"\nONNX Inference Time Statistics:")
    print(f"  Mean:   {times.mean():.2f} ms")
    print(f"  Median: {np.median(times):.2f} ms")
    print(f"  Min:    {times.min():.2f} ms")
    print(f"  Max:    {times.max():.2f} ms")
    print(f"  Std:    {times.std():.2f} ms")


def main(args):
    """Main export function"""
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create model
    print("Creating model...")
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
    print(f"✅ Model loaded")
    
    # Move model to CPU for ONNX export
    model = model.cpu()
    
    # Export to ONNX
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    export_to_onnx(model, args.output)
    
    # Verify ONNX model
    if args.verify:
        verify_onnx_model(model, args.output, torch.device('cpu'))
    
    # Benchmark ONNX model
    if args.benchmark:
        benchmark_onnx(args.output, num_runs=1000)
    
    print(f"\n✅ Export complete! ONNX model saved to: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export GTSRB model to ONNX")
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
        '--output',
        type=str,
        default='models/onnx/gtsrb_model.onnx',
        help='Output path for ONNX model'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify ONNX model outputs match PyTorch'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Benchmark ONNX inference speed'
    )
    
    args = parser.parse_args()
    main(args)
