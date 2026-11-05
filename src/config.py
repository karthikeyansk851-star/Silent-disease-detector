"""Central configuration: paths, hyperparameters, class names."""
from pathlib import Path

# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

HAM10000_IMAGES_DIR = DATA_RAW / "HAM10000_images"
HAM10000_METADATA = DATA_RAW / "HAM10000_metadata.csv"

# --- Classes (HAM10000 diagnosis codes + additions) ---
CLASS_NAMES = [
    "akiec",  # actinic keratoses / intraepithelial carcinoma
    "bcc",    # basal cell carcinoma
    "bkl",    # benign keratosis-like lesions
    "df",     # dermatofibroma
    "mel",    # melanoma
    "normal", # normal skin, no lesion present
    "nv",     # melanocytic nevi
    "scar",   # surgical scar / stitches, not a lesion
    "scc",    # squamous cell carcinoma (invasive)
    "vasc",   # vascular lesions
]
NUM_CLASSES = len(CLASS_NAMES)

# --- Training hyperparameters ---
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 3e-4
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42

MODEL_NAME = "efficientnet_b0"  # any timm model name works
CHECKPOINT_PATH = MODELS_DIR / "best_melanoma_recall_model.pt"

# --- Export ---
ONNX_PATH = MODELS_DIR / "model.onnx"
TFLITE_FP32_PATH = MODELS_DIR / "model_fp32.tflite"
TFLITE_INT8_PATH = MODELS_DIR / "model_int8.tflite"
