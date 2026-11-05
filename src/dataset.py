"""PyTorch Dataset for HAM10000 skin lesion images."""
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from src.config import CLASS_NAMES, HAM10000_IMAGES_DIR, HAM10000_METADATA


class HAM10000Dataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, images_dir: Path = HAM10000_IMAGES_DIR,
                 transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        label_str = row["dx"]

        img_path = self.images_dir / f"{image_id}.jpg"
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.class_to_idx[label_str]
        return image, label


def load_metadata(metadata_path: Path = HAM10000_METADATA) -> pd.DataFrame:
    """Load and lightly clean the HAM10000 metadata CSV."""
    df = pd.read_csv(metadata_path)
    df = df.dropna(subset=["dx", "image_id"])
    return df
