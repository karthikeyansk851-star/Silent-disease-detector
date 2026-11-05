"""
merge_datasets.py
Merges the filtered ISIC 2019 mel/akiec images into your HAM10000 dataset
so training can use a single combined metadata CSV + a single images folder,
matching what HAM10000Dataset expects (image_id, dx columns; f"{image_id}.jpg").
"""

import shutil
from pathlib import Path

import pandas as pd

from src.config import HAM10000_IMAGES_DIR, HAM10000_METADATA, DATA_RAW

# ---------- CONFIG ----------
ISIC_FILTERED_DIR = Path(r"data\isic_extra_filtered")
ISIC_FILTERED_IMG_DIR = ISIC_FILTERED_DIR / "images"
ISIC_FILTERED_METADATA = ISIC_FILTERED_DIR / "isic_extra_metadata.csv"

# Output: combined metadata CSV (original HAM10000_metadata.csv is left untouched)
COMBINED_METADATA_CSV = DATA_RAW / "combined_metadata.csv"
# ---------------------------------


def main():
    # 1. Load existing HAM10000 metadata
    print(f"Loading HAM10000 metadata from {HAM10000_METADATA} ...")
    ham_df = pd.read_csv(HAM10000_METADATA)
    ham_df = ham_df.dropna(subset=["dx", "image_id"])
    ham_df = ham_df[["image_id", "dx"]].copy()
    print(f"HAM10000 rows: {len(ham_df)}")
    print(ham_df["dx"].value_counts())

    # 2. Load filtered ISIC extra metadata
    print(f"\nLoading filtered ISIC metadata from {ISIC_FILTERED_METADATA} ...")
    isic_df = pd.read_csv(ISIC_FILTERED_METADATA)

    # Rename to match HAM10000Dataset's expected columns
    isic_df = isic_df.rename(columns={"isic_id": "image_id", "class_label": "dx"})
    isic_df = isic_df[["image_id", "dx"]].copy()
    print(f"ISIC extra rows: {len(isic_df)}")
    print(isic_df["dx"].value_counts())

    # 3. Copy ISIC images into the same folder HAM10000 images live in
    HAM10000_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    copied, missing, skipped = 0, [], 0

    for image_id in isic_df["image_id"]:
        found = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = ISIC_FILTERED_IMG_DIR / f"{image_id}{ext}"
            if candidate.exists():
                found = candidate
                break
        if found is None:
            missing.append(image_id)
            continue

        # Always copy as .jpg-named target to match HAM10000Dataset's f"{image_id}.jpg" lookup
        dst = HAM10000_IMAGES_DIR / f"{image_id}.jpg"
        if dst.exists():
            skipped += 1
            continue
        if found.suffix.lower() == ".jpg":
            shutil.copy2(found, dst)
        else:
            # Convert non-jpg to jpg so the filename lookup works
            from PIL import Image
            Image.open(found).convert("RGB").save(dst, "JPEG")
        copied += 1

    print(f"\nCopied {copied} new images into {HAM10000_IMAGES_DIR}")
    if skipped:
        print(f"Skipped {skipped} images that already existed at destination")
    if missing:
        print(f"⚠ {len(missing)} images referenced in metadata were not found on disk. "
              f"First few: {missing[:5]}")

    # 4. Combine metadata and save
    combined = pd.concat([ham_df, isic_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["image_id"])
    combined.to_csv(COMBINED_METADATA_CSV, index=False)

    print(f"\nSaved combined metadata to {COMBINED_METADATA_CSV}")
    print(f"Total combined rows: {len(combined)}")
    print("\nFinal combined class distribution:")
    print(combined["dx"].value_counts())


if __name__ == "__main__":
    main()