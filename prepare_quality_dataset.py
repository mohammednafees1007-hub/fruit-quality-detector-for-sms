from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_FRUITBENCH_CACHE = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "datasets--TJIET--FruitBench"
    / "snapshots"
)
QUALITY_CLASSES = ("adulterated", "fresh", "rotten")
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def normalize_quality(stage: str) -> str | None:
    text = re.sub(r"[^a-z0-9]+", " ", stage.lower()).strip()
    if text.startswith("fresh") or text in {"mature", "ripe", "good"}:
        return "fresh"
    if text.startswith("rotten") or text in {"rot", "spoiled", "decayed"}:
        return "rotten"
    if any(token in text for token in ("formalin", "adulter", "damage", "pest")):
        return "adulterated"
    return None


def latest_fruitbench_label_dir(explicit: Path | None) -> Path:
    if explicit:
        root = explicit
        if (root / "label").exists():
            return root / "label"
        if root.name == "label":
            return root
        raise RuntimeError(f"FruitBench path does not contain a label folder: {root}")

    if not DEFAULT_FRUITBENCH_CACHE.exists():
        raise RuntimeError(f"FruitBench cache not found: {DEFAULT_FRUITBENCH_CACHE}")
    candidates = [
        item / "label"
        for item in DEFAULT_FRUITBENCH_CACHE.iterdir()
        if item.is_dir() and (item / "label").exists()
    ]
    if not candidates:
        raise RuntimeError(f"No FruitBench label parquet files found under {DEFAULT_FRUITBENCH_CACHE}")
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return value.strip("_") or "image"


def copy_image_bytes(image_obj: dict, destination: Path) -> bool:
    raw = image_obj.get("bytes") if isinstance(image_obj, dict) else None
    if not raw:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        handle.write(raw)
    try:
        with Image.open(destination) as image:
            image.verify()
        return True
    except Exception:
        destination.unlink(missing_ok=True)
        return False


def split_rows(rows: list[dict], seed: int) -> dict[str, list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["quality"], row["fruit_type"])].append(row)

    rng = random.Random(seed)
    splits = {name: [] for name in SPLIT_RATIOS}
    for group_rows in grouped.values():
        rng.shuffle(group_rows)
        count = len(group_rows)
        train_count = max(1, int(round(count * SPLIT_RATIOS["train"])))
        val_count = max(1, int(round(count * SPLIT_RATIOS["val"])))
        if train_count + val_count >= count:
            train_count = max(1, count - 2)
            val_count = 1 if count >= 3 else 0
        splits["train"].extend(group_rows[:train_count])
        splits["val"].extend(group_rows[train_count:train_count + val_count])
        splits["test"].extend(group_rows[train_count + val_count:])
    for split_rows_value in splits.values():
        rng.shuffle(split_rows_value)
    return splits


def prepare_fruitbench(label_dir: Path, output_dir: Path, seed: int, max_per_quality: int | None) -> dict:
    rows: list[dict] = []
    for parquet_path in sorted(label_dir.glob("*_dataset.parquet")):
        frame = pd.read_parquet(parquet_path)
        for index, record in frame.iterrows():
            quality = normalize_quality(str(record.get("growth_stage", "")))
            if quality is None:
                continue
            rows.append({
                "quality": quality,
                "fruit_type": safe_name(str(record.get("fruit_type") or parquet_path.stem.replace("_dataset", ""))),
                "id": safe_name(str(record.get("id") or f"{parquet_path.stem}_{index}")),
                "image": record.get("image"),
                "stage": str(record.get("growth_stage", "")),
                "source": "FruitBench",
            })

    if max_per_quality:
        rng = random.Random(seed)
        limited = []
        by_quality: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_quality[row["quality"]].append(row)
        for quality, quality_rows in by_quality.items():
            rng.shuffle(quality_rows)
            limited.extend(quality_rows[:max_per_quality])
        rows = limited

    splits = split_rows(rows, seed)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    counts = Counter()
    skipped = 0
    for split_name, split_rows_value in splits.items():
        for row in split_rows_value:
            filename = f"{row['fruit_type']}_{safe_name(row['stage'])}_{row['id']}.jpg"
            destination = output_dir / split_name / row["quality"] / filename
            if copy_image_bytes(row["image"], destination):
                counts[(split_name, row["quality"])] += 1
            else:
                skipped += 1

    for split_name in SPLIT_RATIOS:
        for quality in QUALITY_CLASSES:
            (output_dir / split_name / quality).mkdir(parents=True, exist_ok=True)

    summary = {
        "source": "FruitBench",
        "label_dir": str(label_dir),
        "output_dir": str(output_dir),
        "seed": seed,
        "skipped": skipped,
        "classes": list(QUALITY_CLASSES),
        "splits": {
            split: {quality: counts[(split, quality)] for quality in QUALITY_CLASSES}
            for split in SPLIT_RATIOS
        },
        "mapping": {
            "Mature": "fresh",
            "Rotten": "rotten",
            "Pest-damage": "adulterated",
            "Unripe": "ignored",
        },
    }
    summary_path = output_dir / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare quality-first fruit grading image folders.")
    parser.add_argument("--fruitbench-root", type=Path, default=None, help="FruitBench snapshot root or label folder. Defaults to local Hugging Face cache.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "quality_training" / "imagefolder")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-per-quality", type=int, default=None)
    args = parser.parse_args()

    label_dir = latest_fruitbench_label_dir(args.fruitbench_root)
    summary = prepare_fruitbench(label_dir, args.output_dir, args.seed, args.max_per_quality)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
