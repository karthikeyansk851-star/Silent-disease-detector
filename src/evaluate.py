"""
src/evaluate.py

Runs the trained model on the SAME held-out test split that train.py created
(identical SEED + split logic), and produces:
  1. A full classification report (precision/recall/F1 per class)
  2. A confusion matrix heatmap saved as models/confusion_matrix.png

Now evaluates against the combined HAM10000 + ISIC2019 metadata
(data/raw/combined_metadata.csv), matching train.py's expanded dataset,
and loads best_melanoma_recall_model.pt (the checkpoint chosen for
deployment) instead of the plain best_model.pt.

Usage (from your project root, venv activated):
    python -m src.evaluate
"""

import torch
from torch.utils.data import DataLoader

from src.augmentations import get_eval_transforms
from src.config import (
    CLASS_NAMES,
    BATCH_SIZE,
    DATA_RAW,
    MODEL_NAME,
    MODELS_DIR,
    NUM_CLASSES,
    SEED,
    TEST_SPLIT,
    VAL_SPLIT,
)
from src.dataset import HAM10000Dataset, load_metadata

# Evaluate against the merged dataset (was the HAM10000-only metadata)
COMBINED_METADATA_CSV = DATA_RAW / "combined_metadata.csv"

# Use the melanoma-recall-optimized checkpoint (was CHECKPOINT_PATH / best_model.pt)
EVAL_CHECKPOINT_PATH = MODELS_DIR / "best_melanoma_recall_model.pt"


def get_test_loader():
    """Reproduces the exact same split train.py used, so we evaluate on the
    same held-out test images the model never saw during training."""
    from sklearn.model_selection import train_test_split  # lazy import

    df = load_metadata(COMBINED_METADATA_CSV)

    train_df, temp_df = train_test_split(
        df,
        test_size=(VAL_SPLIT + TEST_SPLIT),
        stratify=df["dx"],
        random_state=SEED,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT),
        stratify=temp_df["dx"],
        random_state=SEED,
    )

    test_ds = HAM10000Dataset(test_df, transform=get_eval_transforms())

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return test_loader


def build_model():
    import timm  # lazy import, matches your train.py pattern

    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
    return model


def evaluate():
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    print("[STEP 1] Rebuilding test split...", flush=True)
    test_loader = get_test_loader()
    print(f"[STEP 1] Test samples: {len(test_loader.dataset)}", flush=True)

    print("[STEP 2] Loading model + checkpoint...", flush=True)
    model = build_model()
    model.load_state_dict(torch.load(EVAL_CHECKPOINT_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print(f"[STEP 2] Model loaded from {EVAL_CHECKPOINT_PATH}", flush=True)

    y_true, y_pred = [], []

    print("[STEP 3] Running inference on test set...", flush=True)
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            y_true.extend(labels.numpy())
            y_pred.extend(preds)

    print("[STEP 3] Inference complete.\n", flush=True)

    # --- Classification report ---
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=2)
    print("=== Per-Class Classification Report ===\n")
    print(report)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = MODELS_DIR / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    # --- Confusion matrix heatmap ---
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Skin Lesion Classifier")
    plt.tight_layout()

    cm_path = MODELS_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)

    print(f"\nSaved confusion matrix to {cm_path}")
    print(f"Saved classification report to {report_path}")


if __name__ == "__main__":
    evaluate()