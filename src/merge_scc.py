import os
import shutil
import pandas as pd

IMAGES_DIR = "data/raw/HAM10000_images"
META_PATH = "data/raw/combined_metadata.csv"
SCC_DIR = "data/isic_new/scc"

# Backup original metadata before touching it
backup_path = META_PATH.replace(".csv", "_backup_before_scc.csv")
shutil.copy(META_PATH, backup_path)
print(f"Backed up metadata to {backup_path}")

meta = pd.read_csv(META_PATH)
print(f"Existing rows: {len(meta)}")

scc_files = [f for f in os.listdir(SCC_DIR) if f.endswith(".jpg")]
copied = 0
new_rows = []

for fname in scc_files:
    image_id = fname.replace(".jpg", "")
    src = os.path.join(SCC_DIR, fname)
    dst = os.path.join(IMAGES_DIR, fname)
    if not os.path.exists(dst):
        shutil.copy(src, dst)
        copied += 1
    new_rows.append({"image_id": image_id, "dx": "scc"})

new_df = pd.DataFrame(new_rows)
combined = pd.concat([meta, new_df], ignore_index=True)
combined.to_csv(META_PATH, index=False)

print(f"Copied {copied} new images into {IMAGES_DIR}")
print(f"New total rows in metadata: {len(combined)}")
print(combined["dx"].value_counts())