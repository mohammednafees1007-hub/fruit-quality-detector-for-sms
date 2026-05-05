# Fruit Quality Detector for SMS

Quality-first FastAPI website for fruit grading.

The main product is quality and grade:

- `Fresh` -> `Grade A`
- `Adulterant / Damaged` -> `Grade B`
- `Rotten` -> `Grade C`

Fruit name, YOLO box, and CLIP source are still shown, but they are secondary context.

## Run

```powershell
cd C:\Users\moham\OneDrive\Documents\FruitQualityWeb_Rebuild
.\run_website.bat
```

Open:

```text
http://127.0.0.1:8000/
```

## Current Pipeline

```text
upload/camera image
  -> optional Camera Robust preprocessing
  -> YOLOv8x fruit box detection
  -> quality model runs on YOLO fruit crop, or full image if no box exists
  -> Keras quality grader predicts fresh/adulterated/rotten
  -> grade mapping returns A/B/C
  -> CLIP ViT-L/14 names the fruit as supporting context
  -> VGG19/MobileNet ImageNet fallbacks support fruit naming only
```

The app refuses grading if `models/fruit_quality_grader.keras` is missing. It does not use a visual quality guess in the prediction path.

## Quality Model

Deployed model:

```text
models/fruit_quality_grader.keras
models/quality_classes.json
models/fruit_quality_grader.metrics.json
```

Current trained model is a MobileNetV2 image classifier trained from the prepared public FruitBench imagefolder dataset. It is deployed because it outperformed the local ConvNeXtTiny run on macro-F1.

Current holdout test metrics:

```text
accuracy: 76.79%
macro-F1: 76.54%
recall adulterated: 71.43%
recall fresh: 91.07%
recall rotten: 67.86%
```

This is a real trained model, not a heuristic. It is usable for demo, but it did not reach the target threshold of 80% accuracy and 70% recall for every class. Better final accuracy needs more formalin/adulterated examples from FruitVision or longer GPU training.

## Prepare Dataset

```powershell
.\.venv\Scripts\python.exe prepare_quality_dataset.py
```

Output:

```text
data/quality_training/imagefolder/train/{adulterated,fresh,rotten}
data/quality_training/imagefolder/val/{adulterated,fresh,rotten}
data/quality_training/imagefolder/test/{adulterated,fresh,rotten}
```

The deterministic split seed is `1337`.

## Train

EfficientNetV2B0 target command:

```powershell
.\.venv\Scripts\python.exe train_quality_backbone.py --backbone efficientnetv2b0 --epochs 12 --fine-tune-epochs 4
```

ConvNeXtTiny comparison command:

```powershell
.\.venv\Scripts\python.exe train_quality_backbone.py --backbone convnexttiny --output models\fruit_quality_grader_convnext.keras --epochs 8 --fine-tune-epochs 0
```

CPU-friendly fallback command used to produce the current deployed model:

```powershell
.\.venv\Scripts\python.exe train_quality_backbone.py --backbone mobilenetv2 --image-size 160 --batch-size 24 --epochs 10 --fine-tune-epochs 0
```

## Health Check

```text
http://127.0.0.1:8000/health
```

`/health` reports `quality_model_ready`, `quality_model_path`, `quality_classes`, and `grading_mode: quality_first`.
