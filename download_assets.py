#!/usr/bin/env python3
"""
Script to download TimesFM model and datasets locally for offline use.
"""

import os
import sys
from pathlib import Path

# Add the timesfm path
sys.path.insert(0, '/scratch/wd04/sm0074/timesfm')

try:
    from huggingface_hub import snapshot_download
    print("✓ huggingface_hub available")
except ImportError:
    print("✗ huggingface_hub not available. Install with: pip install huggingface_hub")
    sys.exit(1)

# Model to download
MODEL_REPO = "google/timesfm-2.0-500m-jax"
LOCAL_MODEL_DIR = "/scratch/wd04/sm0074/timesfm/models"

# Datasets to download
DATASETS = [
    "m1_monthly", "m1_quarterly", "m1_yearly", "m3_monthly", "m3_other",
    "m3_quarterly", "m3_yearly", "m4_quarterly", "m4_yearly",
    "tourism_monthly", "tourism_quarterly", "tourism_yearly",
    "nn5_daily_without_missing", "m5", "nn5_weekly", "traffic", "weather",
    "australian_electricity_demand", "car_parts_without_missing",
    "cif_2016", "covid_deaths", "ercot", "ett_small_15min", "ett_small_1h",
    "exchange_rate", "fred_md", "hospital"
]

def download_model():
    print(f"Downloading model {MODEL_REPO} to {LOCAL_MODEL_DIR}")
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    snapshot_download(repo_id=MODEL_REPO, local_dir=LOCAL_MODEL_DIR)
    print("✓ Model downloaded")

def download_datasets():
    try:
        from gluonts.dataset.repository.datasets import get_dataset
        print("✓ gluonts available")
    except ImportError:
        print("✗ gluonts not available. Install with: pip install gluonts")
        return

    for dataset in DATASETS:
        print(f"Downloading dataset {dataset}")
        try:
            get_dataset(dataset)
            print(f"✓ {dataset} downloaded/cached")
        except Exception as e:
            print(f"✗ Failed to download {dataset}: {e}")

if __name__ == "__main__":
    download_model()
    download_datasets()
    print("All downloads complete!")