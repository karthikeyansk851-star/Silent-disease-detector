import pandas as pd
import os
import shutil

# Paths
CSV_PATH = 'data/raw/combined_metadata.csv'
IMG_DIR = 'data/raw/HAM10000_images'
NEW_DF_DIR = 'data/isic_new/df'
NEW_VASC_DIR = 'data/isic_new/vasc'

# 1. Backup the CSV first
backup_path = CSV_PATH.replace('.csv', '_backup_before_df_vasc_merge.csv')
shutil.copy(CSV_PATH, backup_path)
print(f"Backed up CSV to {backup_path}")

# 2. Load existing metadata
m = pd.read_csv(CSV_PATH)
existing_ids = set(m['image_id'])

new_rows = []
copied = 0
skipped = 0

def merge_class(src_dir, dx_label):
    global copied, skipped
    for fname in os.listdir(src_dir):
        img_id = os.path.splitext(fname)[0]
        if img_id in existing_ids:
            skipped += 1
            continue
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(IMG_DIR, fname)
        shutil.copy(src_path, dst_path)
        new_rows.append({'image_id': img_id, 'dx': dx_label})
        copied += 1

merge_class(NEW_DF_DIR, 'df')
merge_class(NEW_VASC_DIR, 'vasc')

# 3. Append new rows and save
new_df_rows = pd.DataFrame(new_rows)
combined = pd.concat([m, new_df_rows], ignore_index=True)
combined.to_csv(CSV_PATH, index=False)

print(f"Copied {copied} new images, skipped {skipped} duplicates")
print(f"Total dataset size now: {len(combined)}")
print(combined['dx'].value_counts())