#!/usr/bin/env bash
# Downloads the HAM10000 skin lesion dataset from Kaggle into data/raw/.
#
# Prerequisite: place your Kaggle API token at ~/.kaggle/kaggle.json
# (Kaggle account -> Settings -> API -> Create New Token)

set -e

RAW_DIR="$(dirname "$0")/../data/raw"
mkdir -p "$RAW_DIR"

echo "Downloading HAM10000 dataset from Kaggle..."
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p "$RAW_DIR"

echo "Unzipping..."
unzip -o "$RAW_DIR/skin-cancer-mnist-ham10000.zip" -d "$RAW_DIR"

echo "Merging image folders (HAM10000_images_part_1 + part_2)..."
mkdir -p "$RAW_DIR/HAM10000_images"
cp "$RAW_DIR"/HAM10000_images_part_1/*.jpg "$RAW_DIR/HAM10000_images/" 2>/dev/null || true
cp "$RAW_DIR"/HAM10000_images_part_2/*.jpg "$RAW_DIR/HAM10000_images/" 2>/dev/null || true

# Metadata file is usually named HAM10000_metadata.csv already
echo "Done. Images in: $RAW_DIR/HAM10000_images"
echo "Metadata in: $RAW_DIR/HAM10000_metadata.csv"
