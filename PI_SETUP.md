# Raspberry Pi Run Notes

This branch is for running the current FastAPI demo on Raspberry Pi without requiring the full laptop detector stack.

## What was changed for Pi

- Created a `pi` branch from `main`.
- Kept the GitHub fruit-name order the same: CLIP first, then VGG19 ImageNet, then MobileNetV2 ImageNet.
- Downloaded and verified the GitHub default CLIP model `ViT-L/14` locally. The weight file is ignored by git at `models/clip/ViT-L-14.pt`.
- Added a YOLO runtime fallback in `main.py`. If `ultralytics` or `yolov8x.pt` is missing, the API uses the full image as the crop instead of crashing.
- Kept the Keras quality model path unchanged: `models/fruit_quality_grader.keras`.
- Added `requirements-pi.txt` with the dependency versions used to run on this Pi, including the TensorFlow/Keras and CLIP dependencies that worked here.
- Confirmed `.venv/`, `models/clip/`, and large downloaded model files stay out of git.

## Setup on Raspberry Pi

Use Python from Raspberry Pi OS, then create a local virtual environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
```

Install the Pi dependency set:

```bash
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-pi.txt
```

Cache the CLIP model once. This downloads about 890 MB into `models/clip/`:

```bash
.venv/bin/python clip_fruit_predict.py data/test_images/apple.jpg
```

Run the FastAPI app:

```bash
.venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000/
```

## Verification done on Pi

The branch was verified with the included test images through the running API:

| Image | Fruit result | Source | Model |
| --- | --- | --- | --- |
| `data/test_images/apple.jpg` | Apple, confidence 0.9541 | `clip_zero_shot` | `ViT-L/14` |
| `data/test_images/banana.jpg` | Banana, confidence 0.9938 | `clip_zero_shot` | `ViT-L/14` |
| `data/test_images/pomegranate.jpg` | Pomegranate, confidence 0.9931 | `clip_zero_shot` | `ViT-L/14` |

`GET /health` returned healthy with the quality model ready and `ViT-L-14.pt` cached.

## Current Pi behavior

YOLO detection is optional on this branch. When `ultralytics` or `yolov8x.pt` is not available, the backend logs:

```text
YOLO unavailable; using full-image fallback
```

The app still returns fruit name and quality grade. Detection boxes use the full image in this fallback mode.

For a faster production Pi version, replace the laptop detector with a small detector such as YOLOv8n, YOLOv11n, or a Hailo-compiled model, and convert the quality model to TFLite.
