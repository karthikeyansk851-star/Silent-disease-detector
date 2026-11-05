#!/usr/bin/env python3
"""
train.py
========
Silent Disease Detector — Complete Training Script

Fixed version. Original had good ideas (WeightedRandomSampler + FocalLoss
combo, melanoma-recall-specific checkpointing, threshold analysis, AUC-ROC)
but three bugs were corrected:

  1. NO SEEDING (critical) - added set_seed() so runs are reproducible.
     Without this, model init/dropout/batch order are random per run, which
     is exactly what caused wildly different results across earlier runs.
  2. DATA PATHS DIDN'T MATCH THE PROJECT - now imports paths directly from
     src.config (HAM10000_IMAGES_DIR, HAM10000_METADATA) instead of
     hardcoded 'data/...' guesses that don't match this project's actual
     data/raw/ layout.
  3. early_stop_patience=3 + dropout=0.30 + weight_decay=1e-4 is the same
     aggressive combo that already caused underfitting in an earlier run
     (test_acc dropped to 73.85%). Loosened to patience=6, dropout=0.20,
     weight_decay=5e-5 - milder regularization that worked well previously.
  4. Checkpoints are now timestamped per run (so a bad run never overwrites
     a good one), with fixed-name copies kept for evaluate.py/infer.py/
     streamlit_app.py compatibility.

  5. DATASET EXPANDED - metadata_csv now points to data/raw/combined_metadata.csv
     (HAM10000 + filtered, non-overlapping ISIC 2019 mel/akiec images merged via
     merge_datasets.py), instead of the original HAM10000-only metadata. Images
     for both sources live together under HAM10000_IMAGES_DIR, so images_dir is
     unchanged.

- WeightedRandomSampler for true class balancing
- FocalLoss for hard-example mining
- Moderate regularization (dropout 0.20, weight decay 5e-5)
- Melanoma-sensitive checkpointing & threshold tuning
- Lazy imports for Windows stability
- Healthy fitting: early stop on val F1, no confusion matrix

Usage:  python -m src.train
"""
import os
import random
import warnings
from datetime import datetime

# Windows stability fixes
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

from src.config import (
    CLASS_NAMES,
    HAM10000_IMAGES_DIR,
    HAM10000_METADATA,
    DATA_RAW,
    MODELS_DIR,
    SEED,
)

warnings.filterwarnings('ignore', category=UserWarning)

# =============================================================================
# CONFIG
# =============================================================================
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

CONFIG = {
    # Data - now uses the merged HAM10000 + ISIC2019 metadata (was HAM10000_METADATA)
    'metadata_csv': str(DATA_RAW / 'combined_metadata.csv'),
    'images_dir': str(HAM10000_IMAGES_DIR),
    'image_size': 224,
    'num_workers': 0,  # 0 avoids the Windows DLL/threading hang seen before

    # Model
    'model_name': 'efficientnet_b0',
    'pretrained': True,
    'dropout': 0.20,           # was 0.30 - loosened, that value underfit before

    # Training
    'epochs': 25,
    'batch_size': 32,
    'lr': 3e-4,
    'weight_decay': 5e-5,      # was 1e-4 - loosened, same reasoning
    'label_smoothing': 0.1,

    # Balancing
    'use_weighted_sampler': True,
    'use_focal_loss': True,
    'focal_gamma': 2.0,

    # Early stopping
    'early_stop_patience': 6,  # was 3 - too tight, caused early underfitting
    'early_stop_metric': 'val_f1',

    # Scheduler
    'scheduler': 'cosine',
    'T_max': 25,

    # Output
    'output_dir': str(MODELS_DIR),
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}


def set_seed(seed):
    """Seeds Python/NumPy/PyTorch RNGs so a run with identical config
    reliably reproduces the same result. Missing this was the reason
    earlier runs with identical hyperparameters gave very different
    accuracy/F1 numbers."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# DATASET
# =============================================================================
class SkinLesionDataset(Dataset):
    def __init__(self, df, images_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform
        self.image_paths = []
        self.targets = []

        missing = []
        for _, row in self.df.iterrows():
            img_id = row['image_id']
            p = os.path.join(self.images_dir, f"{img_id}.jpg")
            if os.path.exists(p):
                self.image_paths.append(p)
                self.targets.append(CLASS_TO_IDX[row['dx']])
            else:
                missing.append(img_id)

        if missing:
            raise FileNotFoundError(
                f"{len(missing)} image(s) not found under {self.images_dir}. "
                f"First missing: {missing[0]}.jpg. Check src/config.py's "
                f"HAM10000_IMAGES_DIR matches where your images actually are."
            )

        self.targets = np.array(self.targets, dtype=np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.targets[idx]


# =============================================================================
# TRANSFORMS
# =============================================================================
def get_transforms(phase, image_size=224):
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


# =============================================================================
# MODEL (lazy timm)
# =============================================================================
def create_model(num_classes=7, dropout=0.2, pretrained=True):
    import timm
    model = timm.create_model(
        'efficientnet_b0',
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout,
    )
    return model


def build_model(num_classes=None, dropout=None, pretrained=None):
    """Backward-compatible alias. src/gradcam.py imports `build_model` from
    this file (from an earlier version of train.py) - kept here so gradcam.py,
    infer.py, and the Streamlit app keep working without any changes.
    Defaults to this file's CONFIG values when called with no arguments."""
    return create_model(
        num_classes=num_classes if num_classes is not None else NUM_CLASSES,
        dropout=dropout if dropout is not None else CONFIG['dropout'],
        pretrained=pretrained if pretrained is not None else CONFIG['pretrained'],
    )


# =============================================================================
# FOCAL LOSS
# =============================================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets,
            weight=self.alpha,
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        loss = focal_weight * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# =============================================================================
# SAMPLER
# =============================================================================
def get_weighted_sampler(dataset):
    class_counts = np.bincount(dataset.targets, minlength=NUM_CLASSES)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[dataset.targets]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(dataset.targets),
        replacement=True
    )
    return sampler


# =============================================================================
# TRAIN / VAL LOOPS
# =============================================================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    from tqdm import tqdm
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    return total_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return (total_loss / total, correct / total,
            np.array(all_preds), np.array(all_labels), np.array(all_probs))


# =============================================================================
# METRICS (no confusion matrix)
# =============================================================================
def compute_metrics(y_true, y_pred, y_prob, class_names):
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.preprocessing import label_binarize

    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    try:
        macro_auc = roc_auc_score(y_true_bin, y_prob, average='macro', multi_class='ovr')
        weighted_auc = roc_auc_score(y_true_bin, y_prob, average='weighted', multi_class='ovr')
        print(f"\nMacro AUC-ROC:    {macro_auc:.4f}")
        print(f"Weighted AUC-ROC: {weighted_auc:.4f}")
    except Exception:
        macro_auc = 0.0
        weighted_auc = 0.0

    mel_idx = class_names.index('mel')
    mel_mask = (y_true == mel_idx)
    mel_tp = np.sum((y_pred == mel_idx) & mel_mask)
    mel_recall = mel_tp / np.sum(mel_mask) if np.sum(mel_mask) > 0 else 0
    mel_precision = mel_tp / np.sum(y_pred == mel_idx) if np.sum(y_pred == mel_idx) > 0 else 0
    mel_f1 = (2 * mel_precision * mel_recall / (mel_precision + mel_recall)
              if (mel_precision + mel_recall) > 0 else 0)

    print(f"\n{'='*70}")
    print(f"MELANOMA METRICS")
    print(f"{'='*70}")
    print(f"  Sensitivity (Recall): {mel_recall:.4f}    TARGET: >0.90")
    print(f"  Precision:            {mel_precision:.4f}")
    print(f"  F1-Score:             {mel_f1:.4f}")
    print(f"{'='*70}")

    print("\nPer-Class Recall (Sensitivity):")
    for i, cls in enumerate(class_names):
        cls_mask = (y_true == i)
        rec = np.sum((y_pred == i) & cls_mask) / np.sum(cls_mask) if np.sum(cls_mask) > 0 else 0
        marker = "  <<< MELANOMA" if cls == 'mel' else ""
        print(f"  {cls:>6s}: {rec:.4f}{marker}")

    print(f"\n{'='*70}")
    print("MELANOMA THRESHOLD ANALYSIS")
    print("="*70)
    mel_probs = y_prob[:, mel_idx]
    mel_true_bin = mel_mask.astype(int)
    for thresh in [0.10, 0.20, 0.30, 0.50, 0.70]:
        pred_bin = (mel_probs >= thresh).astype(int)
        tp = np.sum((pred_bin == 1) & (mel_true_bin == 1))
        fp = np.sum((pred_bin == 1) & (mel_true_bin == 0))
        fn = np.sum((pred_bin == 0) & (mel_true_bin == 1))
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        print(f"  Threshold {thresh:.2f}:  Sens={sens:.3f}  Prec={prec:.3f}  (FP={fp}, FN={fn})")
    print("="*70)

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, output_dict=True)
    return {
        'accuracy': report['accuracy'],
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_f1': report['weighted avg']['f1-score'],
        'mel_recall': mel_recall,
        'mel_precision': mel_precision,
        'mel_f1': mel_f1,
        'macro_auc': macro_auc,
        'weighted_auc': weighted_auc,
    }


# =============================================================================
# PLOTTING
# =============================================================================
def plot_training_curves(history, save_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Silent Disease Detector — Training History', fontsize=14, fontweight='bold')

    epochs = range(1, len(history['train_loss']) + 1)

    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train')
    ax.plot(epochs, history['val_loss'], 'r-', label='Val')
    ax.set_title('Loss')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, history['train_acc'], 'b-', label='Train')
    ax.plot(epochs, history['val_acc'], 'r-', label='Val')
    ax.set_title('Accuracy')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(epochs, history['val_f1'], 'g-', marker='o')
    ax.set_title('Validation Macro F1')
    ax.set_xlabel('Epoch')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, history['val_mel_recall'], 'r-', marker='s', linewidth=2)
    ax.axhline(0.90, color='green', linestyle='--', label='Clinical Target (0.90)')
    ax.set_title('Melanoma Recall (Validation)')
    ax.set_xlabel('Epoch')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(epochs, history['lr'], 'm-')
    ax.set_title('Learning Rate')
    ax.set_xlabel('Epoch')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    gap = np.array(history['train_acc']) - np.array(history['val_acc'])
    ax.plot(epochs, gap, 'k-', marker='d')
    ax.axhline(0.10, color='orange', linestyle='--', label='Warning (10%)')
    ax.axhline(0.05, color='green', linestyle='--', label='Healthy (5%)')
    ax.set_title('Train-Val Accuracy Gap')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Training curves saved to: {save_path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    set_seed(SEED)

    print("="*70)
    print("Silent Disease Detector")
    print("Healthy Fitting Training")
    print(f"Run ID: {RUN_ID}")
    print("="*70)

    cfg = CONFIG
    device = torch.device(cfg['device'])
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(cfg['output_dir'], exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Load Data
    # -------------------------------------------------------------------------
    print("\n[STEP 1] Loading Data...")
    df = pd.read_csv(cfg['metadata_csv'])
    df = df.dropna(subset=["dx", "image_id"])
    print(f"[DEBUG] Total samples: {len(df)}")
    print("[DEBUG] Class distribution:")
    print(df['dx'].value_counts().sort_index())

    from sklearn.model_selection import train_test_split
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df['dx'], random_state=SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['dx'], random_state=SEED)
    print(f"[DEBUG] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    print(f"[DEBUG] Images directory: {cfg['images_dir']}")

    train_ds = SkinLesionDataset(train_df, cfg['images_dir'], transform=get_transforms('train', cfg['image_size']))
    val_ds   = SkinLesionDataset(val_df,   cfg['images_dir'], transform=get_transforms('val',   cfg['image_size']))
    test_ds  = SkinLesionDataset(test_df,  cfg['images_dir'], transform=get_transforms('val',   cfg['image_size']))

    if cfg['use_weighted_sampler']:
        sampler = get_weighted_sampler(train_ds)
        train_loader = DataLoader(
            train_ds, batch_size=cfg['batch_size'], sampler=sampler,
            num_workers=cfg['num_workers'], pin_memory=(device.type == 'cuda')
        )
        print("[DEBUG] Using WeightedRandomSampler")
        counts = np.bincount(train_ds.targets, minlength=NUM_CLASSES)
        print("[DEBUG] Effective sampling ratio:")
        for i, c in enumerate(CLASS_NAMES):
            ratio = (counts.max() / counts[i]) if counts[i] > 0 else 0
            print(f"         {c:>6s}: {ratio:.1f}x oversample")
    else:
        train_loader = DataLoader(
            train_ds, batch_size=cfg['batch_size'], shuffle=True,
            num_workers=cfg['num_workers'], pin_memory=(device.type == 'cuda')
        )

    val_loader = DataLoader(
        val_ds, batch_size=cfg['batch_size'], shuffle=False,
        num_workers=cfg['num_workers'], pin_memory=(device.type == 'cuda')
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg['batch_size'], shuffle=False,
        num_workers=cfg['num_workers'], pin_memory=(device.type == 'cuda')
    )

    # -------------------------------------------------------------------------
    # 2. Model
    # -------------------------------------------------------------------------
    print("\n[STEP 2] Creating Model...")
    model = create_model(num_classes=NUM_CLASSES, dropout=cfg['dropout'], pretrained=cfg['pretrained'])
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[DEBUG] {cfg['model_name']} | Params: {total_params:,} | Dropout: {cfg['dropout']}")

    # -------------------------------------------------------------------------
    # 3. Loss & Optimizer
    # -------------------------------------------------------------------------
    if cfg['use_focal_loss']:
        criterion = FocalLoss(gamma=cfg['focal_gamma'], label_smoothing=cfg['label_smoothing'])
        print(f"[DEBUG] Loss: FocalLoss (gamma={cfg['focal_gamma']}, smoothing={cfg['label_smoothing']})")
    else:
        class_counts = np.bincount(train_ds.targets, minlength=NUM_CLASSES)
        class_weights = torch.tensor(1.0 / (class_counts + 1e-6), dtype=torch.float32)
        class_weights = class_weights / class_weights.sum() * NUM_CLASSES
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg['label_smoothing'])
        print("[DEBUG] Loss: Weighted CrossEntropyLoss")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

    if cfg['scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['T_max'], eta_min=1e-6)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)

    print(f"[CONFIG] Epochs: {cfg['epochs']} | BS: {cfg['batch_size']} | LR: {cfg['lr']}")
    print(f"[CONFIG] Dropout: {cfg['dropout']} | Weight Decay: {cfg['weight_decay']} | Early Stop: {cfg['early_stop_patience']}")

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    print("\n[STEP 3] Starting Training...")
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_mel_recall': [],
        'lr': []
    }

    best_f1 = 0.0
    best_acc = 0.0
    best_mel_recall = 0.0
    epochs_no_improve = 0

    # Timestamped checkpoint paths - a bad run never overwrites a good one
    ckpt_f1_path = os.path.join(cfg['output_dir'], f'best_model_{RUN_ID}.pt')
    ckpt_acc_path = os.path.join(cfg['output_dir'], f'best_accuracy_model_{RUN_ID}.pt')
    ckpt_mel_path = os.path.join(cfg['output_dir'], f'best_melanoma_recall_model_{RUN_ID}.pt')

    for epoch in range(1, cfg['epochs'] + 1):
        print(f"\n{'='*55}")
        print(f"Epoch {epoch}/{cfg['epochs']}")
        print(f"{'='*55}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, y_pred, y_true, y_prob = validate(model, val_loader, criterion, device)

        metrics = compute_metrics(y_true, y_pred, y_prob, CLASS_NAMES)
        val_f1 = metrics['macro_f1']
        val_mel_recall = metrics['mel_recall']

        if cfg['scheduler'] == 'plateau':
            scheduler.step(val_f1)
        else:
            scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"Val F1:     {val_f1:.4f} | Val Mel Rec: {val_mel_recall:.4f}")
        print(f"LR: {current_lr:.6f}")

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_mel_recall'].append(val_mel_recall)
        history['lr'].append(current_lr)

        gap = train_acc - val_acc
        if gap > 0.15:
            print(f"[WARNING] Overfitting detected! Train-Val gap = {gap:.2%}. Consider stopping early.")

        improved = False

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), ckpt_f1_path)
            print(f"[INFO] Best F1 model saved (Val F1 = {best_f1:.4f})")
            improved = True

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), ckpt_acc_path)
            print(f"[INFO] Best accuracy model saved (Val Acc = {best_acc:.4f})")
            improved = True

        if val_mel_recall > best_mel_recall:
            best_mel_recall = val_mel_recall
            torch.save(model.state_dict(), ckpt_mel_path)
            print(f"[INFO] Best melanoma recall model saved (Val Mel Rec = {best_mel_recall:.4f})")
            improved = True

        if not improved:
            epochs_no_improve += 1
            print(f"[INFO] No improvement for {epochs_no_improve} epoch(s)")
        else:
            epochs_no_improve = 0

        if epochs_no_improve >= cfg['early_stop_patience']:
            print(f"\n[EARLY STOP] No improvement for {cfg['early_stop_patience']} epochs. Stopping.")
            break

    # -------------------------------------------------------------------------
    # 5. Plots
    # -------------------------------------------------------------------------
    curves_path = os.path.join(cfg['output_dir'], f'training_curves_{RUN_ID}.png')
    plot_training_curves(history, curves_path)

    # -------------------------------------------------------------------------
    # 6. Final Test — Best F1 Model
    # -------------------------------------------------------------------------
    print("\n[STEP 4] Testing Best-F1 Model...")
    model.load_state_dict(torch.load(ckpt_f1_path, map_location=device, weights_only=True))
    test_loss, test_acc, y_pred, y_true, y_prob = validate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    compute_metrics(y_true, y_pred, y_prob, CLASS_NAMES)

    # -------------------------------------------------------------------------
    # 7. Final Test — Best Melanoma Recall Model
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("TESTING BEST MELANOMA-RECALL MODEL")
    print("="*70)
    model.load_state_dict(torch.load(ckpt_mel_path, map_location=device, weights_only=True))
    _, _, y_pred_mel, y_true_mel, y_prob_mel = validate(model, test_loader, criterion, device)
    compute_metrics(y_true_mel, y_pred_mel, y_prob_mel, CLASS_NAMES)

    # -------------------------------------------------------------------------
    # Copy timestamped checkpoints to fixed names for evaluate.py/infer.py/
    # streamlit_app.py compatibility, while the timestamped originals stay
    # as a permanent record of this specific run.
    # -------------------------------------------------------------------------
    import shutil
    shutil.copyfile(ckpt_f1_path, os.path.join(cfg['output_dir'], 'best_model.pt'))
    shutil.copyfile(ckpt_acc_path, os.path.join(cfg['output_dir'], 'best_accuracy_model.pt'))
    shutil.copyfile(ckpt_mel_path, os.path.join(cfg['output_dir'], 'best_melanoma_recall_model.pt'))

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Run ID: {RUN_ID}")
    print(f"Best Val F1:      {best_f1:.4f}")
    print(f"Best Val Acc:     {best_acc:.4f}")
    print(f"Best Val Mel Rec: {best_mel_recall:.4f}")
    print(f"\nThis run's artifacts (timestamped, permanent):")
    print(f"  {ckpt_f1_path}")
    print(f"  {ckpt_acc_path}")
    print(f"  {ckpt_mel_path}   <-- USE THIS")
    print(f"  {curves_path}")
    print(f"\nLatest copies (overwritten each run, used by evaluate.py etc):")
    print(f"  {cfg['output_dir']}/best_model.pt")
    print(f"  {cfg['output_dir']}/best_accuracy_model.pt")
    print(f"  {cfg['output_dir']}/best_melanoma_recall_model.pt")
    print("="*70)


if __name__ == '__main__':
    main()