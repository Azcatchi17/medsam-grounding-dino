"""Métriques d'évaluation et utilitaires de chargement d'images."""

from pathlib import Path
import numpy as np
from skimage import io


def load_rgb_image(image_path: str | Path) -> np.ndarray:
    """Charge une image depuis le disque et garantit un format RGB (H, W, 3) en uint8."""
    img = io.imread(str(image_path))
    if len(img.shape) == 2:  # Image en niveaux de gris
        img = np.repeat(img[:, :, None], 3, axis=-1)
    elif img.shape[2] == 4:  # Image RGBA avec canal alpha
        img = img[:, :, :3]
    return img.astype(np.uint8)


def calculate_dice(mask_gt: np.ndarray, mask_pred: np.ndarray, smooth: float = 1e-5) -> float:
    """Calcule le coefficient de Dice (F1-score binaire) entre le masque vérité terrain

    et le masque prédit.
    """
    gt = (mask_gt > 0).astype(np.float32)
    pred = (mask_pred > 0).astype(np.float32)

    intersection = np.sum(gt * pred)
    dice = (2.0 * intersection + smooth) / (np.sum(gt) + np.sum(pred) + smooth)
    return float(dice)