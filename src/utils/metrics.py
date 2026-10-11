"""Evaluation metrics and image loading utilities."""

from pathlib import Path
import numpy as np
from skimage import io


def load_rgb_image(image_path: str | Path) -> np.ndarray:
    """Load an image from disk and ensure standard RGB uint8 format (H, W, 3)."""
    img = io.imread(str(image_path))
    # Expand grayscale images to 3 channels
    if len(img.shape) == 2:
        img = np.repeat(img[:, :, None], 3, axis=-1)
    # Drop alpha channel for RGBA images
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img.astype(np.uint8)


def calculate_dice(mask_gt: np.ndarray, mask_pred: np.ndarray, smooth: float = 1e-5) -> float:
    """Compute the Dice similarity coefficient between ground truth and predicted binary masks."""
    gt = (mask_gt > 0).astype(np.float32)
    pred = (mask_pred > 0).astype(np.float32)

    intersection = np.sum(gt * pred)
    # Smooth term prevents division by zero when both masks are empty
    dice = (2.0 * intersection + smooth) / (np.sum(gt) + np.sum(pred) + smooth)
    return float(dice)