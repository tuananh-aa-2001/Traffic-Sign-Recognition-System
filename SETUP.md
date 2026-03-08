# Quick Setup Guide for GTSRB Traffic Sign Recognition

## Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for training)
- ~5GB free disk space for dataset

## Installation Steps

### 1. Navigate to Project Directory
```bash
cd "c:\Users\Tuan\Desktop\Traffic Sign Recognition System (GTSRB)"
```

### 2. Create Virtual Environment
```bash
python -m venv venv
```

### 3. Activate Virtual Environment
```bash
# Windows
venv\Scripts\activate

# If you see (venv) in your terminal, you're good to go!
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install all required packages including PyTorch, OpenCV, and other dependencies.

### 5. Download Dataset (Automatic)
The dataset will be downloaded automatically when you start training. No manual download needed!

## Quick Start Commands

### Train the Model
```bash
python train.py
```

This will:
- Download GTSRB dataset automatically (~1GB)
- Train the model for up to 50 epochs
- Save the best model to `models/checkpoints/best_model.pth`
- Generate training curves in `results/training_curves.png`

**Training Time**:
- With GPU: 2-3 hours
- Without GPU: 20-30 hours (not recommended)

### Evaluate the Model
After training completes, evaluate on test set:

```bash
python evaluate.py --save_cm --benchmark
```

This will:
- Load the best model
- Evaluate on test set
- Show accuracy metrics
- Save confusion matrix to `results/confusion_matrix.png`
- Benchmark inference speed

### Export to ONNX
For deployment:

```bash
python export_onnx.py --verify --benchmark
```

This will:
- Export model to `models/onnx/gtsrb_model.onnx`
- Verify ONNX outputs match PyTorch
- Benchmark ONNX inference speed

## Monitoring Training

To view training progress in real-time:

```bash
tensorboard --logdir results/tensorboard
```

Then open http://localhost:6006 in your browser.

## Troubleshooting

### CUDA Out of Memory
If you get CUDA out of memory errors, reduce batch size in `configs/config.yaml`:
```yaml
training:
  batch_size: 64  # Reduce from 128
```

### Slow Training on CPU
Training on CPU is very slow. Consider using Google Colab with free GPU:
1. Upload project to Google Drive
2. Open in Colab
3. Enable GPU: Runtime → Change runtime type → GPU

### Dataset Download Fails
If automatic download fails, manually download from:
https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/

Extract to `data/raw/GTSRB/`

## Expected Results

After training, you should see:
- **Test Accuracy**: ~98-99%
- **Inference Time**: ~4ms per image (GPU) or ~15ms (CPU)
- **Model Size**: ~44MB (PyTorch), ~42MB (ONNX)

## Next Steps

1. **Visualize Results**: Check `results/` folder for training curves and confusion matrix
2. **Test on Custom Images**: Modify `evaluate.py` to test on your own traffic sign images
3. **Deploy**: Use the ONNX model for deployment in production systems

## Need Help?

Check the main README.md for detailed documentation, or open an issue on GitHub.

---

**Happy Training! 🚀**
