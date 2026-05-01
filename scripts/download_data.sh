#!/bin/bash
# Download both datasets from Kaggle
# Requires ~/.kaggle/kaggle.json with your API credentials

set -e

echo "Downloading Indiana University OpenI dataset..."
kaggle datasets download raddar/chest-xrays-indiana-university \
    -p data/indiana --unzip

echo "Downloading NIH ChestX-ray14..."
kaggle datasets download nih-chest-xrays/data \
    -p data/nih --unzip

echo "Done. Directory sizes:"
du -sh data/indiana data/nih
