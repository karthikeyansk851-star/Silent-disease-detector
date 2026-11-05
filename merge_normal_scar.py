import os
import shutil
import pandas as pd

IMAGES_DIR = "data/raw/HAM10000_images"
META_PATH = "data/raw/combined_metadata.csv"

SOURCES = {
    "normal": r"data/wounds_raw/Wound_dataset copy/Normal",
    "scar": r"data/wounds_raw/Wound_dataset copy/Surgical Wounds",
}

backup_path = META_PATH.replace(".csv", "_backup_before_normal_scar.csv")
shutil.copy(META_PATH, backup_path)
print(f"Backed up metadata to {backup_path}")

meta = pd.read_csv(META_PATH)
print(f"Existing rows: {len(meta)}")

new_rows = []
total_copied = 0

for dx_label, src_dir in SOURCES.items():
    files = [f for f in os.listdir(src_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    copied = 0
    for fname in files:
        ext = os.path.splitext(fname)[1]
        image_id = f"wound_{dx_label}_{os.path.splitext(fname)[0]}"
        src = os.path.join(src_dir, fname)
        dst = os.path.join(IMAGES_DIR, f"{image_id}{ext}")
        if not os.path.exists(dst):
            shutil.copy(src, dst)
            copied += 1
        new_rows.append({"image_id": image_id, "dx": dx_label})
    print(f"  {dx_label}: copied {copied} new images from {src_dir}")
    total_copied += copied

new_df = pd.DataFrame(new_rows)
combined = pd.concat([meta, new_df], ignore_index=True)
combined.to_csv(META_PATH, index=False)

print(f"\nTotal copied: {total_copied}")
print(f"New total rows in metadata: {len(combined)}")
print(combined["dx"].value_counts())
