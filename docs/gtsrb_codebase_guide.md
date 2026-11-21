# GTSRB Codebase Guide

Welcome to the Traffic Sign Recognition System (GTSRB) codebase! Since this is a structured PyTorch project, understanding where everything lives and how the modules interact is crucial. This guide will walk you through the system from the initial setup to the end goal of model exportation.

## The Big Picture: Where to Start and End

1. **Start Here**: 
   - `src/utils/download_dataset.py` – This is the very first script you run to pull the raw GTSRB data into your local environment.
   - `configs/gtsrb.yaml` (Assuming it exists based on code) – Where all the hyperparameters and paths are defined.
2. **The Core Loop**: 
   - You interact primarily with `scripts/training/train_gtsrb.py` to train the model, and `scripts/auswertung/evaluate_gtsrb.py` to test it.
3. **End Here**: 
   - `scripts/auswertung/export_onnx_gtsrb.py` – Once you have a trained model architecture, this script freezes and exports it to the ONNX format for deployment/inference outside of PyTorch.

---

## Detailed File Connections

Here is how the Python files interact with each other, broken down by logical component.

### 1. Entry Points (The Scripts You Run)

These are the top-level scripts sitting in your root directory.

- **`scripts/training/train_gtsrb.py`**
  - **Purpose**: The main engine for training the model.
  - **Connections**: 
    - Loads the dataset by calling `get_data_loaders()` from `src/data/data_loader.py`.
    - Creates the model by calling `create_model()` from `src/models/resnet_gtsrb.py`.
    - Uses the `Trainer` class from `src/training/trainer.py` to run the active training loop.

- **`scripts/auswertung/evaluate_gtsrb.py`**
  - **Purpose**: Tests a completely trained model against the hold-out test set.
  - **Connections**:
    - Instantiates data and models exactly like `scripts/training/train_gtsrb.py` (via `data_loader.py` and `resnet_gtsrb.py`).
    - Imports `MetricsTracker` and `plot_confusion_matrix` from `src/training/metrics.py` to compute and visualize results.
    - Pulls English translations of the signs via `get_all_class_names()` from `src/utils/class_names.py`.

- **`scripts/auswertung/export_onnx_gtsrb.py`**
  - **Purpose**: Converts the PyTorch `.pth` checkpoint into a deployable `.onnx` model format.
  - **Connections**:
    - Relies heavily on `create_model()` from `src/models/resnet_gtsrb.py` to build the blank model architecture before loading the weights and casting to ONNX.

---

### 2. The Data Layer (`src/data/`)

This directory is responsible for transforming raw `.ppm` files into tensors.

- **`data_loader.py`**
  - **Purpose**: Defines the `GTSRBDataset` class and the `get_data_loaders` function. It splits the directory structure into train/val/test iterators.
  - **Connections**:
    - Deeply connected to **`preprocessing.py`** (to apply CLAHE lighting normalization).
    - Deeply connected to **`augmentation.py`** (to apply random noise, rotation, and scaling during the training split).

- **`preprocessing.py`**
  - **Purpose**: Applies deterministic cleanups: Resizing to 48x48, global ImageNet normalization (`normalize_image`), and adaptive contrast (`CLAHEPreprocessor`).
  - **Connections**: Consumed entirely by `data_loader.py`.

- **`augmentation.py`**
  - **Purpose**: Applies random variations (rotations, translations, scaling, noise) exclusively to the training set to prevent overfitting.
  - **Connections**: Consumed entirely by `data_loader.py`.

---

### 3. The Model Layer (`src/models/`)

- **`resnet_gtsrb.py`**
  - **Purpose**: Defines your Neural Network architectures. It includes a custom `ResNetGTSRB` (ResNet18 modified for tiny 48x48 images by altering the first convolution) and a `LightweightCNN` for faster inference.
  - **Connections**: Exposes the `create_model()` factory function, which is the heart of `scripts/training/train_gtsrb.py`, `scripts/auswertung/evaluate_gtsrb.py`, and `scripts/auswertung/export_onnx_gtsrb.py`.

---

### 4. The Training Logic (`src/training/`)

- **`trainer.py`**
  - **Purpose**: Contains the complex boilerplate for PyTorch training (`Trainer` class). It steps through epochs, performs backpropagation, tracks learning rates with a scheduler, handles Mixed Precision scaling, and saves checkpoints if validation accuracy improves.
  - **Connections**:
    - Consumes the `MetricsTracker` and `EarlyStopping` classes from **`metrics.py`** to decide when to stop training gracefully.

- **`metrics.py`**
  - **Purpose**: All mathematical evaluations live here. It tracks precision/recall/F1 bounds, detects early stopping conditions, and draws physical curves/confusion matrices using matplotlib and seaborn.
  - **Connections**: Used by `scripts/auswertung/evaluate_gtsrb.py` for post-training analytics, and by `trainer.py` step-by-step during training.

---

### 5. The Utilities (`src/utils/`)

These act as helper functions across the entire repository.

- **`download_dataset.py`**
  - **Purpose**: Automates the initial downloading and unzipping of the official GTSRB archives from erda.dk.
  - **Connections**: Standalone script.

- **`class_names.py`**
  - **Purpose**: A hardcoded dictionary mapping `0-42` IDs to English/German strings (e.g., ID 14 = "Stop").
  - **Connections**: Primarily used by `scripts/auswertung/evaluate_gtsrb.py` and `visualization.py` formats logs cleanly.

- **`visualization.py`**
  - **Purpose**: Plots predictions directly bounding the actual images to review what the model gets right/wrong.
  - **Connections**: Uses `get_class_name()` from `class_names.py` to label the images accurately.

---

## Conclusion

If you're new to the codebase, **start by reading `src/data/data_loader.py` and `src/models/resnet_gtsrb.py`**, as these dictate your inputs and your model's capacity. Then look into how `src/training/trainer.py` glues them together during `scripts/training/train_gtsrb.py` execution.
