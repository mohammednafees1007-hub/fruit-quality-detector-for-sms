from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a custom YOLO fruit detector.")
    parser.add_argument("--data", default="data/fruits360-yolo/fruit_yolo.yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project", default="runs/fruit-yolo")
    parser.add_argument("--name", default="yolov8n-fruits")
    parser.add_argument("--output", default="models/custom_fruit_detector.pt")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=10,
        cos_lr=True,
        close_mosaic=10,
    )
    best_model = Path(results.save_dir) / "weights" / "best.pt"
    if not best_model.exists():
        raise RuntimeError(f"Best model was not found: {best_model}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_model, output)
    print(f"Saved custom fruit YOLO model: {output.resolve()}")


if __name__ == "__main__":
    main()
