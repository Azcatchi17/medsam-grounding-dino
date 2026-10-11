"""Evaluate the Oracle baseline (MedSAM with ground-truth bounding boxes) on Kvasir-SEG."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from skimage import io
from tqdm import tqdm

from src.models.medsam_wrapper import MedSAMPredictor
from src.utils.metrics import calculate_dice, load_rgb_image
from src.utils.paths import DATA_DIR, RESULTS_DIR


def evaluate_oracle_kvasir():
    # Setup data and output paths
    kvasir_dir = DATA_DIR / "Kvasir-SEG"
    img_folder = kvasir_dir / "images"
    masks_folder = kvasir_dir / "masks"
    bboxes_file = kvasir_dir / "kavsir_bboxes.json"

    results_folder = RESULTS_DIR / "kvasir"
    results_folder.mkdir(parents=True, exist_ok=True)

    output_masks_folder = results_folder / "oracle_segmentation_masks_kvasir"
    output_masks_folder.mkdir(parents=True, exist_ok=True)

    csv_output_path = results_folder / "dice_oracle_kvasir.csv"

    # Load existing results if already computed
    if csv_output_path.exists():
        print(f"Results already exist at {csv_output_path}.")
        print("Skipping inference and loading CSV file...")
        df = pd.read_csv(csv_output_path)
    else:
        # Load ground truth bounding boxes and map images to their split
        with open(bboxes_file, "r", encoding="utf-8") as f:
            bboxes = json.load(f)

        image_to_split = {}
        for split_name in ["train", "val", "test"]:
            split_file = kvasir_dir / f"{split_name}.json"
            if split_file.exists():
                with open(split_file, "r", encoding="utf-8") as f:
                    coco_data = json.load(f)
                    for img_info in coco_data["images"]:
                        img_id = Path(img_info["file_name"]).stem
                        image_to_split[img_id] = split_name

        # Initialize MedSAM predictor
        predictor = MedSAMPredictor()

        dice_list = []
        print("\nRunning MedSAM Oracle segmentation...")

        # Sort files to ensure deterministic iteration order
        image_files = sorted([f for f in img_folder.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])

        # Main inference loop
        for img_path in tqdm(image_files, desc="MedSAM Oracle"):
            img_id = img_path.stem
            split = image_to_split.get(img_id, "unknown")

            img_3c = load_rgb_image(img_path)
            H, W, _ = img_3c.shape

            # Compute image embedding once per image
            image_embedding, shape = predictor.encode_image(img_3c)

            full_medsam_seg = np.zeros((H, W), dtype=np.uint8)

            # Predict a mask for each bounding box and merge results
            if img_id in bboxes:
                for bbox in bboxes[img_id]["bbox"]:
                    x_min, y_min = bbox["xmin"], bbox["ymin"]
                    x_max, y_max = bbox["xmax"], bbox["ymax"]

                    medsam_seg = predictor.predict_mask_from_box(
                        image_embedding=image_embedding,
                        bbox=[x_min, y_min, x_max, y_max],
                        original_shape=shape,
                    )
                    # Union mask when multiple bounding boxes exist for the same image
                    full_medsam_seg[medsam_seg > 0] = 1

            # Save predicted binary mask as an 8-bit image
            dest_mask_path = output_masks_folder / f"seg_{img_path.name}"
            io.imsave(
                str(dest_mask_path),
                (full_medsam_seg * 255).astype(np.uint8),
                check_contrast=False,
            )

            # Compute Dice score if ground truth mask is available
            mask_path = masks_folder / img_path.name
            if mask_path.exists():
                true_seg = io.imread(str(mask_path))
                if len(true_seg.shape) == 3:
                    true_seg = true_seg[:, :, 0]
                true_seg = (true_seg > 0).astype(np.uint8)

                dice_score = calculate_dice(true_seg, full_medsam_seg)

                dice_list.append({
                    "image_id": img_id,
                    "split": split,
                    "dice": dice_score,
                })

        # Save all results to CSV
        df = pd.DataFrame(dice_list)
        df.to_csv(csv_output_path, index=False)
        print(f"\nResults saved to {csv_output_path}")

    # Display evaluation summary
    print("\nOverall Oracle Dice:")
    print(df["dice"].describe())

    if "split" in df.columns:
        print("\nOracle Dice by split:")
        print(df.groupby("split")["dice"].describe())


if __name__ == "__main__":
    evaluate_oracle_kvasir()