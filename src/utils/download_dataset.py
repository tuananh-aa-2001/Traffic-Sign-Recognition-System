"""
Automated GTSRB Dataset Download and Extraction
Downloads the German Traffic Sign Recognition Benchmark dataset
"""

import os
import zipfile
import requests
from tqdm import tqdm
from pathlib import Path


def download_file(url: str, destination: str, chunk_size: int = 8192) -> None:
    """
    Download a file with progress bar
    
    Args:
        url: URL to download from
        destination: Local file path to save to
        chunk_size: Download chunk size in bytes
    """
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    with open(destination, 'wb') as file, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            progress_bar.update(size)


def extract_zip(zip_path: str, extract_to: str) -> None:
    """
    Extract a zip file
    
    Args:
        zip_path: Path to zip file
        extract_to: Directory to extract to
    """
    print(f"Extracting {os.path.basename(zip_path)}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extraction complete!")


def dataset_ready(data_dir: str = "data/raw") -> bool:
    """True if training images, test images and test labels are all in place."""
    gtsrb_dir = os.path.join(data_dir, "GTSRB")
    return (
        os.path.isdir(os.path.join(gtsrb_dir, "Final_Training", "Images"))
        and os.path.isdir(os.path.join(gtsrb_dir, "Final_Test", "Images"))
        and os.path.exists(os.path.join(gtsrb_dir, "GT-final_test.csv"))
    )


def download_gtsrb(data_dir: str = "data/raw") -> None:
    """
    Download and extract the GTSRB dataset
    
    Args:
        data_dir: Directory to save the dataset
    """
    # Create data directory
    os.makedirs(data_dir, exist_ok=True)
    
    # GTSRB dataset URLs (from official Kaggle mirror)
    # Note: These are the most reliable public sources
    train_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip"
    test_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip"
    test_labels_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip"
    
    # Download paths
    train_zip = os.path.join(data_dir, "GTSRB_Final_Training_Images.zip")
    test_zip = os.path.join(data_dir, "GTSRB_Final_Test_Images.zip")
    test_labels_zip = os.path.join(data_dir, "GTSRB_Final_Test_GT.zip")
    
    # Download training data
    if not os.path.exists(train_zip):
        print("Downloading GTSRB Training Images...")
        download_file(train_url, train_zip)
    else:
        print("Training images already downloaded.")
    
    # Download test data
    if not os.path.exists(test_zip):
        print("Downloading GTSRB Test Images...")
        download_file(test_url, test_zip)
    else:
        print("Test images already downloaded.")
    
    # Download test labels
    if not os.path.exists(test_labels_zip):
        print("Downloading GTSRB Test Labels...")
        download_file(test_labels_url, test_labels_zip)
    else:
        print("Test labels already downloaded.")
    
    # Extract files (each part separately, so an interrupted run can resume)
    gtsrb_dir = os.path.join(data_dir, "GTSRB")
    if not os.path.isdir(os.path.join(gtsrb_dir, "Final_Training", "Images")):
        extract_zip(train_zip, data_dir)
    if not os.path.isdir(os.path.join(gtsrb_dir, "Final_Test", "Images")):
        extract_zip(test_zip, data_dir)

    # The label archive extracts GT-final_test.csv into data_dir itself,
    # but the data loader expects it inside GTSRB/. Move it there.
    labels_csv = os.path.join(gtsrb_dir, "GT-final_test.csv")
    if not os.path.exists(labels_csv):
        extract_zip(test_labels_zip, data_dir)
        extracted_csv = os.path.join(data_dir, "GT-final_test.csv")
        if os.path.exists(extracted_csv):
            os.replace(extracted_csv, labels_csv)
    if not os.path.exists(labels_csv):
        raise FileNotFoundError(f"Test labels missing after extraction: {labels_csv}")
    print("\n✅ GTSRB dataset ready.")
    
    # Print dataset info
    print("\n" + "="*60)
    print("GTSRB Dataset Information:")
    print("="*60)
    print(f"Dataset location: {os.path.abspath(data_dir)}")
    print(f"Training images: GTSRB/Final_Training/Images/")
    print(f"Test images: GTSRB/Final_Test/Images/")
    print(f"Test labels: GT-final_test.csv")
    print(f"Number of classes: 43")
    print(f"Training samples: ~39,209")
    print(f"Test samples: 12,630")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download GTSRB dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw",
        help="Directory to save the dataset"
    )
    
    args = parser.parse_args()
    download_gtsrb(args.data_dir)
