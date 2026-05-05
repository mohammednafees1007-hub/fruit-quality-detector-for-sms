from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml


DATASET_REPO = "https://github.com/fruits-360/fruits-360-100x100.git"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

BASE_LABELS = (
    "cabbage white", "cabbage red", "cactus fruit", "passion fruit",
    "pomelo sweetie", "melon piel de sapo", "ginger root", "peanut shell",
    "caju seed", "bean pod", "blackberry", "carambola", "cherimoya",
    "gooseberry", "apple", "apricot", "almonds", "avocado", "banana",
    "beetroot", "blueberry", "cantaloupe", "cauliflower", "cherry",
    "chestnut", "clementine", "cocos", "corn", "carrot", "cucumber",
    "dates", "eggplant", "fig", "ginger", "granadilla", "grape",
    "grapefruit", "guava", "hazelnut", "huckleberry", "kaki", "kiwi",
    "kohlrabi", "kumquats", "lemon", "lime", "limes", "lychee",
    "mandarine", "mango", "mangostan", "maracuja", "mulberry",
    "nectarine", "nut", "onion", "orange", "papaya", "peach", "pear",
    "pepino", "pepper", "physalis", "pineapple", "pistachio", "pitahaya",
    "plum", "pomegranate", "potato", "quince", "rambutan", "raspberry",
    "redcurrant", "salak", "strawberry", "tamarillo", "tangelo", "tomato",
    "walnut", "watermelon", "zucchini",
)

LABEL_ALIASES = {
    "almonds": "almond",
    "cabbage red": "red cabbage",
    "cabbage white": "cabbage",
    "caju seed": "cashew seed",
    "cocos": "coconut",
    "dates": "date",
    "ginger": "ginger root",
    "kumquats": "kumquat",
    "limes": "lime",
    "mangostan": "mangosteen",
    "maracuja": "passion fruit",
    "melon piel de sapo": "melon",
    "peanut shell": "peanut",
    "pitahaya": "dragon fruit",
    "pomelo sweetie": "pomelo",
}


def ensure_dataset(data_dir: Path) -> None:
    if (data_dir / "Training").exists() and (data_dir / "Test").exists():
        print(f"Fruits-360 dataset already exists: {data_dir}")
        return
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", DATASET_REPO, str(data_dir)], check=True)


def normalize_folder_name(folder_name: str) -> str | None:
    cleaned = folder_name.lower().replace("_", " ")
    cleaned = re.sub(r"\br\d*\b", " ", cleaned)
    cleaned = re.sub(r"\d+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    for base in sorted(BASE_LABELS, key=len, reverse=True):
        if cleaned == base or cleaned.startswith(base + " "):
            return LABEL_ALIASES.get(base, base).replace(" ", "_")
    return None


def find_images(split_dir: Path) -> list[tuple[Path, str]]:
    samples: list[tuple[Path, str]] = []
    skipped: Counter[str] = Counter()
    for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        label = normalize_folder_name(class_dir.name)
        if not label:
            skipped[class_dir.name] += 1
            continue
        for image_path in class_dir.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((image_path, label))
    if skipped:
        print("Skipped unknown folders:")
        print(json.dumps(dict(skipped), indent=2))
    if not samples:
        raise RuntimeError(f"No images found under {split_dir}")
    return samples


def cap_per_class(samples: list[tuple[Path, str]], max_per_class: int | None, seed: int) -> list[tuple[Path, str]]:
    if not max_per_class:
        return samples
    rng = random.Random(seed)
    grouped: dict[str, list[Path]] = defaultdict(list)
    for image_path, label in samples:
        grouped[label].append(image_path)
    capped: list[tuple[Path, str]] = []
    for label, paths in grouped.items():
        rng.shuffle(paths)
        capped.extend((path, label) for path in paths[:max_per_class])
    return sorted(capped, key=lambda item: str(item[0]))


def infer_yolo_bbox(image_path: Path) -> tuple[float, float, float, float]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    mask = ((saturation > 28) | (value < 245)).astype("uint8") * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype="uint8"))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype="uint8"))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.5, 0.5, 1.0, 1.0
    x, y, w_box, h_box = cv2.boundingRect(max(contours, key=cv2.contourArea))
    pad = max(2, int(max(w_box, h_box) * 0.06))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(width, x + w_box + pad)
    y2 = min(height, y + h_box + pad)
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    return (x1 + box_w / 2) / width, (y1 + box_h / 2) / height, box_w / width, box_h / height


def copy_split(samples: list[tuple[Path, str]], output_dir: Path, split_name: str, label_to_index: dict[str, int]) -> Counter[str]:
    image_output = output_dir / "images" / split_name
    label_output = output_dir / "labels" / split_name
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    for source_path, label in samples:
        counts[label] += 1
        stem = f"{label}_{counts[label]:06d}"
        target_image = image_output / f"{stem}{source_path.suffix.lower()}"
        target_label = label_output / f"{stem}.txt"
        shutil.copy2(source_path, target_image)
        cx, cy, w_box, h_box = infer_yolo_bbox(source_path)
        target_label.write_text(f"{label_to_index[label]} {cx:.6f} {cy:.6f} {w_box:.6f} {h_box:.6f}\n", encoding="utf-8")
    return counts


def write_dataset_yaml(output_dir: Path, labels: list[str]) -> Path:
    yaml_path = output_dir / "fruit_yolo.yaml"
    yaml_data = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {index: label.replace("_", " ") for index, label in enumerate(labels)},
    }
    yaml_path.write_text(yaml.safe_dump(yaml_data, sort_keys=False), encoding="utf-8")
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a YOLO-format fruit dataset from Fruits-360.")
    parser.add_argument("--source", default="data/fruits360-100x100")
    parser.add_argument("--output", default="data/fruits360-yolo")
    parser.add_argument("--max-per-class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    ensure_dataset(source)
    train_samples = cap_per_class(find_images(source / "Training"), args.max_per_class, args.seed)
    val_cap = None if not args.max_per_class else max(1, min(80, args.max_per_class // 5))
    val_samples = cap_per_class(find_images(source / "Test"), val_cap, args.seed)
    labels = sorted({label for _, label in train_samples} | {label for _, label in val_samples})
    label_to_index = {label: index for index, label in enumerate(labels)}
    if args.clean and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    train_counts = copy_split(train_samples, output, "train", label_to_index)
    val_counts = copy_split(val_samples, output, "val", label_to_index)
    yaml_path = write_dataset_yaml(output, labels)
    print(f"Prepared YOLO dataset: {output.resolve()}")
    print(f"Classes: {len(labels)}")
    print(f"Train images: {sum(train_counts.values())}")
    print(f"Validation images: {sum(val_counts.values())}")
    print(f"Dataset YAML: {yaml_path.resolve()}")


if __name__ == "__main__":
    main()
