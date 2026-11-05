"""Simple heuristic pre-filter to reject images that likely aren't a close-up lesion photo.

Not a learned model — uses basic color/contour heuristics to catch obvious
non-lesion inputs (scars, wide shots, random skin, backgrounds) before they
reach the classifier.
"""
import cv2
import numpy as np
from PIL import Image


def has_lesion(pil_image: Image.Image, min_area_frac=0.01, max_area_frac=0.6):
    """Returns (bool, debug_info_dict).

    True if the image likely contains a centered, lesion-sized dark/irregular blob.
    False if no such region is found (likely a scar, wide shot, or non-lesion image).
    """
    img = np.array(pil_image.convert("RGB"))
    h, w = img.shape[:2]
    img_area = h * w

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Otsu threshold to separate darker lesion-like regions from lighter skin
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Clean up noise
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return False, {"reason": "no_contours_found"}

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    area_frac = area / img_area

    x, y, cw, ch = cv2.boundingRect(largest)
    aspect_ratio = max(cw, ch) / max(min(cw, ch), 1)
    cx, cy = x + cw / 2, y + ch / 2
    center_dist_x = abs(cx - w / 2) / w
    center_dist_y = abs(cy - h / 2) / h

    debug = {
        "area_frac": round(area_frac, 4),
        "center_dist_x": round(center_dist_x, 3),
        "center_dist_y": round(center_dist_y, 3),
        "aspect_ratio": round(aspect_ratio, 2),
    }

    if area_frac < min_area_frac or area_frac > max_area_frac:
        debug["reason"] = "area_out_of_range"
        return False, debug

    if center_dist_x > 0.35 or center_dist_y > 0.35:
        debug["reason"] = "not_centered"
        return False, debug
    if aspect_ratio > 2.5:
        debug["reason"] = "too_elongated_likely_not_lesion"
        return False, debug

    debug["reason"] = "passed"
    return True, debug


if __name__ == "__main__":
    import sys
    img = Image.open(sys.argv[1]).convert("RGB")
    result, debug = has_lesion(img)
    print(f"has_lesion: {result}")
    print("debug:", debug)