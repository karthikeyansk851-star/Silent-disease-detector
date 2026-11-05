"""Grad-CAM heatmap generation so health workers can see why a region was flagged."""
import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.augmentations import get_eval_transforms
from src.config import CHECKPOINT_PATH, CLASS_NAMES, MODEL_NAME, NUM_CLASSES
from src.train import build_model


def load_trained_model(device="cpu"):
    model = build_model()
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    model.to(device).eval()
    return model


def get_target_layer(model):
    """Returns a late conv block with good spatial resolution for localization.
    blocks[-2] (second-to-last MBConv block) tends to localize on the actual
    lesion better than conv_head, which is a 1x1 pointwise conv right before
    pooling and has less spatial detail to work with."""
    if hasattr(model, "blocks"):
        return [model.blocks[-2]]
    return [model.conv_head]


def generate_gradcam(model, pil_image: Image.Image, device="cpu"):
    """Returns (predicted_class_name, confidence, heatmap_overlay_rgb_uint8, top3_list, all_probs).

    top3_list is a list of (class_name, confidence) tuples, sorted highest first.
    all_probs is a dict {class_name: probability} for EVERY class, so callers
    can check a specific class's probability (e.g. melanoma) even when it
    doesn't rank in the top 3 - important once you have 10+ classes, since a
    top-3-only view can silently miss a class that's clinically important but
    not the single highest score.
    """
    pil_image_rgb = pil_image.convert("RGB")

    transform = get_eval_transforms()
    input_tensor = transform(pil_image_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = int(probs.argmax())
        confidence = float(probs[pred_idx])

        all_probs = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

        top3_probs, top3_idx = torch.topk(probs, min(3, len(CLASS_NAMES)))
        top3_list = [
            (CLASS_NAMES[int(idx)], float(p))
            for p, idx in zip(top3_probs, top3_idx)
        ]

    target_layers = get_target_layer(model)
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)

    # Explicitly explain the predicted class, and smooth the map to reduce
    # noisy/off-target activations that don't correspond to the lesion.
    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets,
                         aug_smooth=True, eigen_smooth=True)[0]

    rgb_img = np.array(
        pil_image_rgb.resize((input_tensor.shape[-1], input_tensor.shape[-2]))
    ).astype(np.float32) / 255.0

    overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    return CLASS_NAMES[pred_idx], confidence, overlay, top3_list, all_probs


def save_debug_overlay(overlay: np.ndarray, path: str = "debug_gradcam_output.png"):
    """Optional helper: save the raw overlay array straight to disk so you can
    inspect it in a file explorer, bypassing any camera/screen photo artifacts.
    """
    Image.fromarray(overlay).save(path)
    return path
