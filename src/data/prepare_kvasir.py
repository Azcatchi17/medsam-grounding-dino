import json
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.utils.paths import DATA_DIR


def convert_kvasir_to_coco_split(input_json_path: Path, output_dir: Path):
    """Effectue un split reproductible (800 train, 100 val, 100 test) et convertit les annotations

    de Kvasir-SEG au format COCO standard pour Grounding DINO et l'évaluation.
    """
    if not input_json_path.exists():
        print(f"Erreur : fichier source introuvable à {input_json_path}")
        print("Assurez-vous d'avoir extrait le zip sous data/Kvasir-SEG/")
        return

    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Ordre identique à l'implémentation d'origine pour préserver le split avec random_state=42
    image_keys = list(data.keys())

    # 1. Séparation Train (800) vs Rest (200)
    train_keys, rest_keys = train_test_split(
        image_keys, train_size=0.8, random_state=42, shuffle=True
    )

    # 2. Séparation Val (100) vs Test (100)
    val_keys, test_keys = train_test_split(
        rest_keys, train_size=0.5, random_state=42, shuffle=True
    )

    def build_coco_dict(keys_subset):
        coco_dict = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "polyp"}],
        }
        ann_id = 1
        for i, img_name in enumerate(keys_subset, start=1):
            img_info = data[img_name]
            coco_dict["images"].append({
                "id": i,
                "file_name": f"{img_name}.jpg",
                "width": img_info["width"],
                "height": img_info["height"],
            })
            for bbox in img_info["bbox"]:
                w = bbox["xmax"] - bbox["xmin"]
                h = bbox["ymax"] - bbox["ymin"]
                coco_dict["annotations"].append({
                    "id": ann_id,
                    "image_id": i,
                    "category_id": 1,
                    "bbox": [float(bbox["xmin"]), float(bbox["ymin"]), float(w), float(h)],
                    "area": float(w * h),
                    "iscrowd": 0,
                })
                ann_id += 1
        return coco_dict

    output_dir.mkdir(parents=True, exist_ok=True)
    splits = [
        ("train.json", train_keys),
        ("val.json", val_keys),
        ("test.json", test_keys),
    ]

    for filename, keys in splits:
        dest_file = output_dir / filename
        with open(dest_file, "w", encoding="utf-8") as f:
            json.dump(build_coco_dict(keys), f, indent=4)
        print(f"Fichier créé : {dest_file} ({len(keys)} images)")


if __name__ == "__main__":
    kvasir_dir = DATA_DIR / "Kvasir-SEG"
    input_bboxes = kvasir_dir / "kavsir_bboxes.json"

    convert_kvasir_to_coco_split(input_bboxes, kvasir_dir)