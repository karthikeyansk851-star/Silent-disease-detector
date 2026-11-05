"""Single-image inference with Grad-CAM overlay. Used by the Streamlit demo."""
from PIL import Image

from src.gradcam import generate_gradcam, load_trained_model

_model = None


def get_model():
    global _model
    if _model is None:
        _model = load_trained_model()
    return _model


def predict(pil_image: Image.Image):
    """Returns (predicted_class_name, confidence, heatmap_overlay_rgb_uint8, top3_list, all_probs).

    all_probs is a dict {class_name: probability} for every class, so callers
    can check a specific class (e.g. melanoma) even when it isn't in top3.
    """
    model = get_model()
    return generate_gradcam(model, pil_image)


if __name__ == "__main__":
    import sys

    img_path = sys.argv[1]
    image = Image.open(img_path).convert("RGB")
    label, confidence, _, top3, all_probs = predict(image)
    print(f"Prediction: {label} (confidence={confidence:.2%})")
    print("Top-3:", top3)
    print(f"Melanoma probability: {all_probs.get('mel', 0):.2%}")
