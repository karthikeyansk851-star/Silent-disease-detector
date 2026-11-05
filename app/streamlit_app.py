"""Streamlit demo: upload a skin lesion image, get a prediction + Grad-CAM overlay.

Run: streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.infer import predict  # noqa: E402
from src.lesion_filter import has_lesion  # noqa: E402

st.set_page_config(page_title="Silent Disease Detector", page_icon=":stethoscope:")

st.title(":stethoscope: Silent Disease Detector - Skin Lesion Screening")
st.caption(
    "Prototype for low-resource early detection via smartphone imaging. "
    "Not a medical diagnosis - for demo purposes only."
)

CLASS_DESCRIPTIONS = {
    "nv": "Melanocytic nevus (benign mole)",
    "mel": "Melanoma (malignant, serious)",
    "bcc": "Basal cell carcinoma (malignant)",
    "scc": "Squamous cell carcinoma (malignant, invasive)",
    "akiec": "Actinic keratosis / Bowen's disease (precancerous)",
    "bkl": "Benign keratosis-like lesion",
    "df": "Dermatofibroma (benign)",
    "vasc": "Vascular lesion",
    "normal": "Normal skin (no lesion detected)",
    "scar": "Surgical scar / stitches (not a lesion)",
}

NON_LESION_CLASSES = {"normal", "scar"}

# Melanoma gets its own dedicated threshold check against its FULL probability
# (not just top-3), since with 10 classes melanoma can have a clinically
# meaningful probability without being the single top-ranked class.
MEL_THRESHOLD = 0.20

# SCC/BCC still use the top-3-based check - their recall is already strong
# (SCC ~0.95-0.98), so a top-3 check is sufficient for them.
OTHER_MALIGNANT_CLASSES = {"scc", "bcc"}
ESCALATION_THRESHOLD = 0.20

CONFIDENCE_THRESHOLD = 0.70

uploaded_file = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded image", use_column_width=True)

    lesion_detected, filter_debug = has_lesion(image)

    if not lesion_detected:
        st.warning(
            "This image may not show a clear, close-up lesion "
            f"(reason: {filter_debug.get('reason', 'unknown')}). "
            "Results below may be unreliable - consider retaking the photo "
            "with better lighting, closer framing on the lesion, and less "
            "background clutter."
        )

    with st.spinner("Analyzing image..."):
        label, confidence, overlay, top3, all_probs = predict(image)

    with col2:
        st.image(overlay, caption="Grad-CAM: what the model focused on", use_column_width=True)

    if lesion_detected:
        if label in NON_LESION_CLASSES:
            st.success(
                f"No lesion detected - top prediction: {CLASS_DESCRIPTIONS.get(label, label)} "
                f"({confidence:.1%})."
            )
        else:
            mel_prob = all_probs.get("mel", 0.0)
            escalated_other = [
                (cls, conf) for cls, conf in top3
                if cls in OTHER_MALIGNANT_CLASSES and conf >= ESCALATION_THRESHOLD
            ]

            findings = []
            if mel_prob >= MEL_THRESHOLD:
                findings.append(("mel", mel_prob))
            findings.extend(escalated_other)

            if findings:
                names = ", ".join(f"{CLASS_DESCRIPTIONS.get(c, c)} ({p:.1%})" for c, p in findings)
                st.error(
                    f"⚠ Possible malignant finding: {names}. "
                    "Please consult a dermatologist promptly for a proper evaluation."
                )
            elif confidence < CONFIDENCE_THRESHOLD:
                st.info(
                    f"Low-to-moderate confidence prediction ({confidence:.1%}). "
                    "Consider a clearer photo or consult a dermatologist for a proper evaluation."
                )

        st.markdown("**Top predictions:**")
        for cls, conf in top3:
            desc = CLASS_DESCRIPTIONS.get(cls, cls)
            st.write(f"- **{cls}** ({desc}) — {conf:.1%}")

        if "mel" not in [c for c, _ in top3]:
            st.caption(f"(Melanoma probability: {all_probs.get('mel', 0):.1%} — not in top 3)")

    st.warning(
        "This is a research prototype, not a certified diagnostic tool. "
        "Please consult a qualified healthcare professional for any medical concerns."
    )
else:
    st.info("Upload an image to get started.")
