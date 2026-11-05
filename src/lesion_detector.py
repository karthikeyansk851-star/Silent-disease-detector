"""
Lesion presence pre-filter.

Checks whether an uploaded image likely contains a distinct skin lesion
(mole, spot, patch) before sending it to the classifier. Uses classical
CV (no deep learning) — fast, explainable, no training data needed.

Usage:
    from lesion_detector import has_lesion

    result = has_lesion("path/to/image.jpg")
    if not result["detected"]:
        print("No clear lesion found:", result["reason"])
    else:
        print("Lesion region found, proceed to classification")
        print("Bounding box:", result["bbox"])
"""

import cv2
import numpy as np


def has_lesion(image_path, min_area_ratio=0.01, max_area_ratio=0.85,
               contrast_threshold=15):
    """
    Detect whether a distinct lesion-like region exists in the image.

    Args:
        image_path: path to the image file
        min_area_ratio: smallest fraction of image area a lesion can occupy
        max_area_ratio: largest fraction of image area a lesion can occupy
                         (avoids treating the whole frame as "the lesion")
        contrast_threshold: min color/intensity difference from surrounding
                             skin to be considered a distinct region

    Returns:
        dict with keys:
            detected (bool)
            reason (str) - explanation if not detected
            bbox (tuple or None) - (x, y, w, h) of the largest candidate region
            confidence (float) - rough heuristic score 0-1
    """
    img = cv2.imread(image_path)
    if img is None:
        return {"detected": False, "reason": "Could not read image", "bbox": None, "confidence": 0.0}

    h, w = img.shape[:2]
    img_area = h * w

    # Convert to LAB color space — better for detecting skin discoloration
    # than RGB, since it separates lightness from color.
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Smooth to reduce noise from skin texture/hair
    blurred = cv2.GaussianBlur(l_channel, (15, 15), 0)

    # Estimate "background skin" tone as the median of the image
    median_val = np.median(blurred)

    # Threshold: pixels that differ significantly from the median skin tone
    diff = np.abs(blurred.astype(np.int16) - int(median_val))
    _, thresh = cv2.threshold(diff.astype(np.uint8), contrast_threshold, 255, cv2.THRESH_BINARY)

    # Clean up small noise / hair artifacts
    kernel = np.ones((7, 7), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {"detected": False, "reason": "No distinct region found — image appears to be uniform skin",
                "bbox": None, "confidence": 0.0}

    # Find largest candidate region
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    area_ratio = area / img_area

    if area_ratio < min_area_ratio:
        return {"detected": False, "reason": f"Detected region too small ({area_ratio:.1%} of image)",
                "bbox": None, "confidence": area_ratio / min_area_ratio}

    if area_ratio > max_area_ratio:
        return {"detected": False, "reason": "No localized region — variation spans nearly the whole image "
                                              "(likely uneven lighting, not a lesion)",
                "bbox": None, "confidence": 0.0}

    x, y, bw, bh = cv2.boundingRect(largest)
    # Rough confidence: how "compact"/circular the region is (lesions tend to be blob-like)
    perimeter = cv2.arcLength(largest, True)
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
    confidence = float(np.clip(circularity, 0.0, 1.0))

    return {
        "detected": True,
        "reason": "Lesion-like region detected",
        "bbox": (x, y, bw, bh),
        "confidence": confidence,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python lesion_detector.py <image_path>")
        sys.exit(1)

    result = has_lesion(sys.argv[1])
    print(result)