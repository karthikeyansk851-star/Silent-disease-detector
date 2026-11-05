import os
import pandas as pd

existing_meta = pd.read_csv("data/raw/combined_metadata.csv")
existing_ids = set(existing_meta["image_id"])

scc_dir = "data/isic_new/scc"
scc_files = [f.replace(".jpg", "") for f in os.listdir(scc_dir) if f.endswith(".jpg")]

overlap = [f for f in scc_files if f in existing_ids]
print(f"Total SCC images downloaded: {len(scc_files)}")
print(f"Overlapping with existing dataset: {len(overlap)}")
print(f"New unique SCC images to add: {len(scc_files) - len(overlap)}")
