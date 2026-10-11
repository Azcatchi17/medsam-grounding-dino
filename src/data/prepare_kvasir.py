"""Prepare Kvasir-SEG dataset by splitting and converting annotations to COCO format."""

import json
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.utils.paths import DATA_DIR


def convert_kvasir_to_coco_split(input_json_path: Path, output_dir: Path):
    """Split Kvasir-SEG dataset (80% train, 10% val, 10% test) and convert
    bounding box annotations into standard COCO JSON format.
    """
    # Input file verification
    if not input_json_path.exists():
        print(f"Error: input file not found at {input_json_path}")
        print("Please check that Kvasir-SEG dataset is extracted in data/Kvasir-SEG/")
        return

    # Load dataset annotations
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_keys = list(data.keys())

    # Dataset split: fixed random_state to ensure reproducible splits
    train_keys, rest_keys = train_test_split(
        image_keys, train_size=0.8, random_state=42, shuffle=True
    )
    val_keys, test_keys = train_test_split(
        rest_keys, train_size=0.5, random_state=42, shuffle=True
    )

    # Convert subset to standard COCO format
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
                # Convert [xmin, ymin, xmax, ymax] to COCO [xmin, ymin, width, height]
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

    # Export COCO JSON files for each split
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
        print(f"Generated: {dest_file} ({len(keys)} images)")


if __name__ == "__main__":
    kvasir_dir = DATA_DIR / "Kvasir-SEG"
    input_bboxes = kvasir_dir / "kavsir_bboxes.json"

    convert_kvasir_to_coco_split(input_bboxes, kvasir_dir)