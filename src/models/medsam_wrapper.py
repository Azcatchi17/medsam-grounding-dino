"""
Wrapper d'inférence pour MedSAM (Medical Segment Anything Model).

Ce module adapte l'implémentation officielle de MedSAM (Wang et al., 2024)
basée sur l'architecture SAM de Meta (Kirillov et al., 2023).

Référence officielle MedSAM :
    Dépôt : https://github.com/bowang-lab/MedSAM
    Article : Wang, B., et al. "Segment Anything in Medical Images",
              Nature Communications (2024).
"""

import os
import warnings
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry
from skimage import transform
from src.utils.paths import CHECKPOINTS_DIR

DEFAULT_MEDSAM_CHECKPOINT = CHECKPOINTS_DIR / "medsam_vit_b.pth"

class MedSAMPredictor:
    """
    Wrapper pour l'encodage d'images et la prédiction de masques avec MedSAM.

    L'architecture s'appuie sur le backbone ViT-B de Segment Anything (Meta AI),
    tandis que les poids et le flux d'inférence par boîte sont issus de MedSAM.
    """
    def __init__(self, checkpoint_path: str = DEFAULT_MEDSAM_CHECKPOINT, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Le fichier de poids {checkpoint_path} est introuvable. "
                f"Veuillez le placer dans le dossier checkpoints/."
            )

        print(f"Chargement de MedSAM ({checkpoint_path}) sur {self.device}...")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="segment_anything")
            self.model = sam_model_registry["vit_b"](checkpoint=str(checkpoint_path))

        self.model = self.model.to(self.device)
        self.model.eval()

    def _medsam_inference(self, image_embedding: torch.Tensor, box_1024: np.ndarray, H: int, W: int) -> np.ndarray:
        """
        Inférence guidée par invite géométrique.

        Adapté fidèlement de la fonction `medsam_inference` du dépôt officiel MedSAM
        (bowang-lab/MedSAM).
        """
        box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=self.device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :]  # Dimension (B, 1, 4)

        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
        low_res_logits, _ = self.model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        low_res_pred = torch.sigmoid(low_res_logits)
        low_res_pred = low_res_pred.squeeze().cpu().numpy()

        medsam_seg = cv2.resize(
            low_res_pred,
            (W, H),
            interpolation=cv2.INTER_CUBIC
        )
        medsam_seg = (medsam_seg > 0.5).astype(np.uint8)
        return medsam_seg

    def encode_image(self, img_3c: np.ndarray):
        """
        Prétraitement et extraction des caractéristiques visuelles.

        Reprend la normalisation min-max et le redimensionnement cubique 1024x1024
        recommandés par le pipeline d'inférence de MedSAM.
        """
        H, W, _ = img_3c.shape

        # Redimensionnement 1024x1024 selon la méthode de MedSAM
        img_1024 = transform.resize(
            img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)

        # Normalisation min-max
        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        )

        img_1024_tensor = (
            torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        )

        with torch.no_grad():
            image_embedding = self.model.image_encoder(img_1024_tensor)

        return image_embedding, (H, W)

    def predict_mask_from_box(self, image_embedding: torch.Tensor, bbox: list, original_shape: tuple) -> np.ndarray:
        """Génère le masque binaire 2D correspondant à une boîte englobante.

        Args:
            image_embedding (torch.Tensor): Caractéristiques de l'image extraites
                par le ViT encoder (shape: [1, 256, 64, 64]).
            bbox (list): Coordonnées de la boîte [xmin, ymin, xmax, ymax] dans le
                repère d'origine de l'image.
            original_shape (tuple): Dimensions réelles (H, W) de l'image
                d'origine.

        Returns:
            np.ndarray: Masque binaire uint8 (0 ou 1) aux dimensions (H, W).
        """
        H, W = original_shape
        x_min, y_min, x_max, y_max = bbox

        box_np = np.array([[x_min, y_min, x_max, y_max]])
        box_1024 = box_np / np.array([W, H, W, H]) * 1024

        with torch.no_grad():
            mask = self._medsam_inference(image_embedding, box_1024, H, W)

        return mask