
from typing import Dict, Any, Optional
import cv2
import numpy as np
from pathlib import Path

def detect_cracks(image_path: str, out_mask_path: Optional[str]=None) -> Dict[str, Any]:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    thin = closed  # fallback if ximgproc is not available
    crack_pixels = int((thin>0).sum())
    area = thin.size
    density = crack_pixels / float(area)
    length_est = crack_pixels / 2.0
    if out_mask_path:
        Path(out_mask_path).parent.mkdir(parents=True, exist_ok=True)
        import imageio.v2 as imageio
        imageio.imwrite(out_mask_path, thin)
    return {
        "crack_pixels": crack_pixels,
        "image_area": int(area),
        "crack_density": round(density,6),
        "length_est_px": round(float(length_est),2),
        "mask_saved_to": out_mask_path if out_mask_path else None
    }
