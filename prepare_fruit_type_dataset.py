from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
FRUITS360_CACHE = HF_CACHE / "datasets--PedroSampaio--fruits-360" / "snapshots"
FRUITBENCH_CACHE = HF_CACHE / "datasets--TJIET--FruitBench" / "snapshots"
QUALITY_SPLITS = {"train": 0.78, "val": 0.10, "test": 0.12}


ALIASES = {
    "carambula": "carambola",
    "cocos": "coconut",
    "corn husk": "corn",
    "cucumber ripe": "cucumber",
    "dates": "date",
    "grape blue": "grape",
    "grape pink": "grape",
    "grape white": "grape",
    "grapefruit pink": "grapefruit",
    "grapefruit white": "grapefruit",
    "limes": "lime",
    "mangostan": "mangosteen",
    "maracuja": "passion fruit",
    "melon piel de sapo": "melon",
    "pitahaya red": "dragon fruit",
    "pomelo sweetie": "pomelo",
    "tomato not ripened": "tomato",
}

PREFIXES = (
    "apple", "apricot", "avocado", "banana", "beetroot", "blueberry",
    "cactus fruit", "cantaloupe", "cauliflower", "cherry", "chestnut",
    "clementine", "corn", "cucumber", "eggplant", "fig", "ginger root",
    "granadilla", "grape", "grapefruit", "guava", "hazelnut",
    "huckleberry", "kaki", "kiwi", "kohlrabi", "kumquat", "lemon",
    "lime", "lychee", "mandarine", "mango", "mangosteen", "melon",
    "mulberry", "nectarine", "nut", "onion", "orange", "papaya",
    "passion fruit", "peach", "pear", "pepino", "pepper", "physalis",
    "pineapple", "plum", "pomegranate", "potato", "quince", "rambutan",
    "raspberry", "redcurrant", "salak", "strawberry", "tamarillo",
    "tangelo", "tomato", "walnut", "watermelon",
)


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return value.strip("_") or "image"


def normalize_fruit_name(name: str) -> str | None:
    text = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
    text = re.sub(r"\s+", " ", text)
    if text in ALIASES:
        return ALIASES[text].replace(" ", "_")
    for prefix in sorted(PREFIXES, key=len, reverse=True):
        if text == prefix or text.startswith(prefix + " "):
            return ALIASES.get(prefix, prefix).replace(" ", "_")
    return None


def latest_snapshot(root: Path) -> Path:
    candidates = [path for path in root.iterdir() if path.is_dir()]
    if not candidates:
        raise RuntimeError(f"No Hugging Face snapshot found under {root}")
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def fruits360_label_names(parquet_path: Path) -> list[str]:
    schema = pq.read_schema(parquet_path)
    metadata = schema.metadata or {}
    payload = json.loads(metadata[b"huggingface"].decode("utf-8"))
    return payload["info"]["features"]["label"]["names"]


def image_bytes(image_obj) -> bytes | None:
    if isinstance(image_obj, dict):
        return image_obj.get("bytes")
    return None


def add_fruits360_rows(rows: list[dict], snapshot: Path) -> None:
    data_dir = snapshot / "data"
    train_path = next(data_dir.glob("train-*.parquet"))
    test_path = next(data_dir.glob("test-*.parquet"))
    names = fruits360_label_names(train_path)
    for parquet_path, source_split in ((train_path, "trainval"), (test_path, "test")):
        frame = pd.read_parquet(parquet_path)
        for index, record in frame.iterrows():
            raw_label = names[int(record["label"])]
            fruit = normalize_fruit_name(raw_label)
            raw = image_bytes(record["image"])
            if fruit and raw:
                rows.append({
                    "fruit": fruit,
                    "source": "fruits360",
                    "source_split": source_split,
                    "id": f"{parquet_path.stem}_{index}",
                    "bytes": raw,
                })


def add_fruitbench_rows(rows: list[dict], label_dir: Path) -> None:
    for parquet_path in sorted(label_dir.glob("*_dataset.parquet")):
        frame = pd.read_parquet(parquet_path)
        for index, record in frame.iterrows():
            fruit = normalize_fruit_name(str(record.get("fruit_type") or parquet_path.stem.replace("_dataset", "")))
            raw = image_bytes(record.get("image"))
            if fruit and raw:
                rows.append({
                    "fruit": fruit,
                    "source": "fruitbench",
                    "source_split": "mixed",
                    "id": safe_name(str(record.get("id") or f"{parquet_path.stem}_{index}")),
                    "bytes": raw,
                })


def split_rows(rows: list[dict], seed: int, max_train: int, max_val: int, max_test: int) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["fruit"]].append(row)

    splits = {"train": [], "val": [], "test": []}
    for fruit, fruit_rows in grouped.items():
        rng.shuffle(fruit_rows)
        explicit_test = [row for row in fruit_rows if row["source_split"] == "test"]
        trainval = [row for row in fruit_rows if row["source_split"] != "test"]
        rng.shuffle(explicit_test)
        rng.shuffle(trainval)
        val_count = min(max_val, max(1, int(len(trainval) * QUALITY_SPLITS["val"])))
        train_count = min(max_train, max(1, len(trainval) - val_count))
        test_count = min(max_test, max(1, len(explicit_test) or int(len(fruit_rows) * QUALITY_SPLITS["test"])))
        splits["val"].extend(trainval[:val_count])
        splits["train"].extend(trainval[val_count:val_count + train_count])
        if explicit_test:
            splits["test"].extend(explicit_test[:test_count])
        else:
            splits["test"].extend(trainval[val_count + train_count:val_count + train_count + test_count])

    for split_rows_value in splits.values():
        rng.shuffle(split_rows_value)
    return splits


def write_image(raw: bytes, path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        path.unlink(missing_ok=True)
        return False


def prepare_dataset(output_dir: Path, seed: int, max_train: int, max_val: int, max_test: int) -> dict:
    rows: list[dict] = []
    fruits360_snapshot = latest_snapshot(FRUITS360_CACHE)
    fruitbench_snapshot = latest_snapshot(FRUITBENCH_CACHE)
    add_fruits360_rows(rows, fruits360_snapshot)
    add_fruitbench_rows(rows, fruitbench_snapshot / "label")
    splits = split_rows(rows, seed, max_train, max_val, max_test)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    counts = Counter()
    skipped = 0
    for split, split_rows_value in splits.items():
        for row in split_rows_value:
            target = output_dir / split / row["fruit"] / f"{row['source']}_{safe_name(row['id'])}.jpg"
            if write_image(row["bytes"], target):
                counts[(split, row["fruit"])] += 1
            else:
                skipped += 1

    classes = sorted({row["fruit"] for row in rows})
    summary = {
        "sources": ["PedroSampaio/fruits-360", "TJIET/FruitBench"],
        "output_dir": str(output_dir),
        "seed": seed,
        "classes": classes,
        "class_count": len(classes),
        "skipped": skipped,
        "splits": {
            split: {fruit: counts[(split, fruit)] for fruit in classes if counts[(split, fruit)]}
            for split in ("train", "val", "test")
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a broad fruit-name dataset from Hugging Face cached datasets.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "fruit_type_training" / "imagefolder")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-train-per-class", type=int, default=320)
    parser.add_argument("--max-val-per-class", type=int, default=70)
    parser.add_argument("--max-test-per-class", type=int, default=90)
    args = parser.parse_args()
    print(json.dumps(prepare_dataset(args.output_dir, args.seed, args.max_train_per_class, args.max_val_per_class, args.max_test_per_class), indent=2))


if __name__ == "__main__":
    main()
