"""Shared image preprocessing for visual model training and inference.

Aspect-ratio preservation:
    Frame extraction, training-feature generation, and inference must produce the
    same tensor layout. The legacy path stretched every frame directly to
    224x224, which distorts portrait phone video. This module letterboxes the
    frame into the target square, preserving geometry with symmetric padding.
"""

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


def letterbox_bgr(
    image_bgr: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    pad_value: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Resize while preserving aspect ratio and center-pad to ``target_size``.

    Returns the padded BGR image plus geometry metadata. Padding is applied
    only when needed; square inputs are resized without padding.
    """
    if image_bgr is None or image_bgr.size == 0 or len(image_bgr.shape) != 3:
        raise ValueError(f"Invalid image for letterbox: {None if image_bgr is None else image_bgr.shape}")

    target_w, target_h = int(target_size[0]), int(target_size[1])
    if target_w <= 0 or target_h <= 0:
        raise ValueError(f"Invalid target size: {target_size}")

    src_h, src_w = image_bgr.shape[:2]
    if src_h <= 0 or src_w <= 0:
        raise ValueError(f"Invalid source dimensions: {src_w}x{src_h}")

    scale = min(target_w / float(src_w), target_h / float(src_h))
    content_w = max(1, min(target_w, int(round(src_w * scale))))
    content_h = max(1, min(target_h, int(round(src_h * scale))))

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image_bgr, (content_w, content_h), interpolation=interpolation)

    canvas = np.full((target_h, target_w, 3), pad_value, dtype=image_bgr.dtype)
    pad_x = (target_w - content_w) // 2
    pad_y = (target_h - content_h) // 2
    canvas[pad_y : pad_y + content_h, pad_x : pad_x + content_w] = resized

    meta = {
        "source_width": src_w,
        "source_height": src_h,
        "content_width": content_w,
        "content_height": content_h,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "scale": round(float(scale), 6),
        "target_size": [target_w, target_h],
    }
    return canvas, meta


def normalize_bgr_to_chw(
    image_bgr: np.ndarray,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> np.ndarray:
    """Convert a BGR uint8 image to normalized float32 CHW."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    for c in range(3):
        rgb[:, :, c] = (rgb[:, :, c] - mean[c]) / std[c]
    return np.transpose(rgb, (2, 0, 1)).astype(np.float32)


def preprocess_letterboxed(
    image_bgr: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Letterbox, normalize, and return CHW plus geometry metadata."""
    boxed, meta = letterbox_bgr(image_bgr, target_size=target_size)
    return normalize_bgr_to_chw(boxed, mean=mean, std=std), meta
