"""Inference wrapper for MedSAM (Medical Segment Anything Model).

Handles image feature extraction and bounding-box prompted mask generation.

Adapted from the official MedSAM repository (bowang-lab/MedSAM, Wang et al., 2024)
built on top of Meta's Segment Anything (SAM) architecture.
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
    """Wrapper to load MedSAM, compute image embeddings, and predict segmentation masks."""

    def __init__(self, checkpoint_path: str = DEFAULT_MEDSAM_CHECKPOINT, device: str = None):
        # Set compute device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Verify weights file existence
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint file not found: {checkpoint_path}. "
                f"Please place it in the checkpoints/ directory."
            )

        print(f"Loading MedSAM ({checkpoint_path}) on {self.device}...")

        # Suppress PyTorch weights_only warning from external segment_anything loader
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="segment_anything")
            self.model = sam_model_registry["vit_b"](checkpoint=str(checkpoint_path))

        self.model = self.model.to(self.device)
        self.model.eval()

    def _medsam_inference(self, image_embedding: torch.Tensor, box_1024: np.ndarray, H: int, W: int) -> np.ndarray:
        """Run decoder inference from image embeddings and a single scaled bounding box."""
        # Format box tensor to (B, 1, 4) required by SAM prompt encoder
        box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=self.device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :]

        # Encode bounding box prompt
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )

        # Decode mask logits
        # multimask_output=False returns a single definitive mask for the targeted lesion
        low_res_logits, _ = self.model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # Convert logits to probabilities
        low_res_pred = torch.sigmoid(low_res_logits)
        low_res_pred = low_res_pred.squeeze().cpu().numpy()

        # Upscale low-resolution prediction (256x256) back to original image dimensions
        medsam_seg = cv2.resize(
            low_res_pred,
            (W, H),
            interpolation=cv2.INTER_CUBIC
        )

        # Threshold probabilities at 0.5 to get binary segmentation
        medsam_seg = (medsam_seg > 0.5).astype(np.uint8)
        return medsam_seg

    def encode_image(self, img_3c: np.ndarray):
        """Preprocess an input image and compute its visual embeddings using the ViT backbone."""
        H, W, _ = img_3c.shape

        # ViT backbone requires a fixed 1024x1024 input resolution
        img_1024 = transform.resize(
            img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)

        # Min-max intensity normalization with epsilon to prevent division by zero
        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        )

        # Convert HWC numpy array to BCHW tensor
        img_1024_tensor = (
            torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        )

        # Compute image embedding via ViT encoder
        with torch.no_grad():
            image_embedding = self.model.image_encoder(img_1024_tensor)

        return image_embedding, (H, W)

    def predict_mask_from_box(self, image_embedding: torch.Tensor, bbox: list, original_shape: tuple) -> np.ndarray:
        """Generate a binary segmentation mask for a given bounding box [xmin, ymin, xmax, ymax].

        Args:
            image_embedding (torch.Tensor): Feature embedding from the ViT image encoder.
            bbox (list): Bounding box coordinates [xmin, ymin, xmax, ymax] in original image space.
            original_shape (tuple): Original image resolution (H, W).

        Returns:
            np.ndarray: Binary mask of shape (H, W) with values in {0, 1}.
        """
        H, W = original_shape
        x_min, y_min, x_max, y_max = bbox

        # Rescale coordinates to the 1024x1024 embedding space
        box_np = np.array([[x_min, y_min, x_max, y_max]])
        box_1024 = box_np / np.array([W, H, W, H]) * 1024

        # Run prompt-guided segmentation
        with torch.no_grad():
            mask = self._medsam_inference(image_embedding, box_1024, H, W)

        return mask