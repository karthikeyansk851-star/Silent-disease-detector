"""
filter_isic.py
Filters the downloaded ISIC 2019 collection down to mel/akiec images only,
excludes any images that overlap with HAM10000 (Collection 212), and copies
the matching files into your existing dataset structure.
"""

import pandas as pd
import shutil
from pathlib import Path

# ---------- CONFIG — adjust these paths to match your project ----------
ISIC_FULL_DIR = Path(r"data\isic_extra_full")
ISIC_METADATA_CSV = ISIC_FULL_DIR / "metadata.csv"

# Where your final filtered images should go (adjust to your training pipeline's expected structure)
OUTPUT_DIR = Path(r"data\isic_extra_filtered")
OUTPUT_IMG_DIR = OUTPUT_DIR / "images"
OUTPUT_METADATA_CSV = OUTPUT_DIR / "isic_extra_metadata.csv"

# If you have a list of HAM10000 ISIC IDs already (e.g. from Collection 212 metadata),
# point this at that CSV. If you don't have it yet, set to None and the script will
# skip overlap filtering (you can re-run once you download it).
HAM10000_METADATA_CSV = Path(r"data\ham10000_reference\metadata.csv")

# Column names (confirmed from your isic_2019_metadata.csv inspection)
DIAGNOSIS_COL = "diagnosis_3"
IMAGE_ID_COL = "isic_id"  # standard ISIC metadata column name for the image identifier

# Target diagnosis values -> your model's class labels
DIAGNOSIS_TO_CLASS = {
    "Melanoma, NOS": "mel",
    "Melanoma in situ": "mel",
    "Melanoma Invasive": "mel",
    "Solar or actinic keratosis": "akiec",
}
# -------------------------------------------------------------------------


def find_image_file(isic_id: str, search_dir: Path) -> Path | None:
    """ISIC downloads are usually named <isic_id>.jpg, but check a couple of extensions."""
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = search_dir / f"{isic_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def main():
    print(f"Reading metadata from {ISIC_METADATA_CSV} ...")
    df = pd.read_csv(ISIC_METADATA_CSV, low_memory=False)
    print(f"Total rows: {len(df)}")

    if DIAGNOSIS_COL not in df.columns:
        raise SystemExit(
            f"Column '{DIAGNOSIS_COL}' not found. Available columns:\n{list(df.columns)}"
        )

    # 1. Filter to only the diagnoses we want
    mask = df[DIAGNOSIS_COL].isin(DIAGNOSIS_TO_CLASS.keys())
    filtered = df.loc[mask].copy()
    print(f"Rows matching target diagnoses: {len(filtered)}")
    print(filtered[DIAGNOSIS_COL].value_counts())

    # 2. Map to your model's class labels
    filtered["class_label"] = filtered[DIAGNOSIS_COL].map(DIAGNOSIS_TO_CLASS)

    # 3. Exclude HAM10000 overlap, if a reference CSV is provided
    if HAM10000_METADATA_CSV is not None and Path(HAM10000_METADATA_CSV).exists():
        ham_df = pd.read_csv(HAM10000_METADATA_CSV, low_memory=False)
        ham_ids = set(ham_df[IMAGE_ID_COL].astype(str))
        before = len(filtered)
        filtered = filtered[~filtered[IMAGE_ID_COL].astype(str).isin(ham_ids)]
        print(f"Removed {before - len(filtered)} rows overlapping with HAM10000")
    else:
        print("⚠ No HAM10000 reference CSV provided — skipping overlap filtering. "
              "Run isic collection download for Collection 212 later and re-run this "
              "script with HAM10000_METADATA_CSV set, to be safe before training.")

    # 4. Copy matching image files
    OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0
    missing = []

    for _, row in filtered.iterrows():
        isic_id = str(row[IMAGE_ID_COL])
        src = find_image_file(isic_id, ISIC_FULL_DIR)
        if src is None:
            missing.append(isic_id)
            continue
        dst = OUTPUT_IMG_DIR / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        copied += 1

    print(f"Copied {copied} images to {OUTPUT_IMG_DIR}")
    if missing:
        print(f"⚠ {len(missing)} image files were not found (check filenames/extensions). "
              f"First few missing IDs: {missing[:5]}")

    # 5. Save the filtered metadata for merging into your training pipeline
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(OUTPUT_METADATA_CSV, index=False)
    print(f"Saved filtered metadata to {OUTPUT_METADATA_CSV}")

    print("\nFinal class distribution:")
    print(filtered["class_label"].value_counts())


if __name__ == "__main__":
    main()
