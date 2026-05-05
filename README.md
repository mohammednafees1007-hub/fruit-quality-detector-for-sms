# Fruit Quality Detector for SMS

Quality-first fruit grading website built with FastAPI, TensorFlow/Keras, YOLOv8x, and CLIP.

Repository: <https://github.com/mohammednafees1007-hub/fruit-quality-detector-for-sms>

## Table of Contents

- [Objective](#objective)
- [Introduction and Motivation](#introduction-and-motivation)
- [Problem Statement](#problem-statement)
- [Methodology](#methodology)
- [System Components](#system-components)
- [Workflow](#workflow)
- [Algorithm](#algorithm)
- [Implementation Details](#implementation-details)
- [Project Structure](#project-structure)
- [Installation and Running](#installation-and-running)
- [API Endpoints](#api-endpoints)
- [Training Pipeline](#training-pipeline)
- [Results and Discussion](#results-and-discussion)
- [Limitations](#limitations)
- [Conclusion](#conclusion)
- [Future Enhancements](#future-enhancements)
- [References](#references)

## Objective

The objective of this project is to build a practical web-based fruit quality grading system for SMS. The system analyzes a fruit image from upload or camera input and returns:

- `Quality`: `Fresh`, `Adulterant`, or `Rotten`
- `Grade`: `A`, `B`, or `C`
- `Fruit name`: supporting context from the fruit naming model
- `Annotated image`: image output with detection box and prediction label

The project is quality-first. Fruit detection and fruit naming are supporting features, while quality and grade are the main output.

## Introduction and Motivation

Fruit quality inspection is commonly done manually. Manual inspection is slow, inconsistent, and depends on the experience of the person checking the fruit. In real-world use, fruit quality can be affected by freshness, decay, physical damage, pest damage, and possible chemical adulteration.

This project uses computer vision to make fruit grading more systematic. A user can upload an image or use a camera, and the system predicts the quality class and grade. This helps demonstrate how machine learning can support food quality inspection, retail sorting, agricultural workflows, and student-level smart monitoring systems.

The motivation for the final design is:

- Quality and grading are the actual product requirement.
- Fruit name detection is useful, but it should not decide the quality.
- A trained image classifier is more honest than a hand-written visual guess.
- If the quality model is missing, the backend must refuse grading instead of guessing.
- The system should be easy to run locally for presentation.

## Problem Statement

Given an input fruit image, classify the fruit into one of three quality categories and map that quality to a grade.

| Quality Class | Display Label | Grade | Meaning |
| --- | --- | --- | --- |
| `fresh` | Fresh | A | Fruit appears healthy and suitable |
| `adulterated` | Adulterant | B | Fruit appears damaged, pest-affected, or adulteration-like |
| `rotten` | Rotten | C | Fruit appears spoiled or rotten |

The application must support:

- Image upload
- Camera capture
- Camera Robust Mode for phone-screen and low-quality captures
- YOLO-based fruit region detection
- Keras-based quality classification
- CLIP-based fruit naming
- Clear API responses for frontend and debugging

## Methodology

The system follows a multi-stage computer vision pipeline.

1. Input image is captured from upload or camera.
2. Optional Camera Robust Mode improves the image when the camera feed is noisy.
3. YOLOv8x detects fruit-like object regions.
4. The quality model receives the YOLO crop when a fruit box is found. If no useful box is found, it uses the full image.
5. A trained Keras classifier predicts `fresh`, `adulterated`, or `rotten`.
6. The predicted quality class is mapped to grade `A`, `B`, or `C`.
7. CLIP ViT-L/14 predicts the fruit name as secondary context.
8. VGG19 and MobileNetV2 ImageNet models are available only as fruit-name fallbacks.
9. The backend returns JSON plus an annotated image for the website.

Important design rule:

CLIP, YOLO, VGG19, and MobileNetV2 do not decide fruit quality. Only the trained Keras quality model decides quality and grade.

## System Components

| Component | File / Model | Purpose |
| --- | --- | --- |
| FastAPI backend | `main.py` | Serves website, `/health`, and `/detect` |
| Embedded website UI | `main.py` | Upload, camera, robust mode toggle, result panel |
| Quality classifier | `models/fruit_quality_grader.keras` | Main model for Fresh / Adulterant / Rotten |
| Quality class metadata | `models/quality_classes.json` | Stores class order used by the model |
| Quality metrics | `models/fruit_quality_grader.metrics.json` | Stores training and test results |
| YOLO detector | `yolov8x.pt` | Detects fruit object boxes |
| CLIP classifier | `clip_fruit_predict.py` | Predicts fruit name using ViT-L/14 first |
| Dataset preparation | `prepare_quality_dataset.py` | Builds train/val/test folders |
| Quality training | `train_quality_backbone.py` | Trains EfficientNetV2, ConvNeXtTiny, or MobileNetV2 |
| Custom YOLO training | `train_custom_yolo_fruits.py` | Future custom fruit detector training |
| Fruits-360 YOLO prep | `prepare_fruits360_yolo.py` | Converts Fruits-360-style data to YOLO format |
| Launcher | `run_website.bat` | Starts the website with one command |

## Workflow

```text
User upload / camera frame
        |
        v
Optional Camera Robust Mode
        |
        v
YOLOv8x fruit box detection
        |
        v
Select quality input:
    - use best YOLO crop if available
    - otherwise use full image
        |
        v
Keras quality classifier
        |
        v
Quality class:
    fresh / adulterated / rotten
        |
        v
Grade mapping:
    Fresh -> A
    Adulterant / Damaged -> B
    Rotten -> C
        |
        v
CLIP fruit name prediction as secondary context
        |
        v
Website result:
    Grade, Quality, Confidence, Fruit Name, Annotated Image
```

## Algorithm

### Detection and Grading Algorithm

```text
Input: image file, robust_camera flag
Output: quality, grade, confidence, fruit name, annotated image

1. Read image bytes from request.
2. Decode image using OpenCV.
3. If robust_camera is enabled:
      a. Try to crop phone-screen-like region.
      b. Apply contrast normalization.
      c. Apply brightness correction.
      d. Apply sharpening.
4. Run YOLOv8x on the processed image.
5. If YOLO returns fruit boxes:
      a. Select the best fruit box using area and YOLO confidence.
      b. Crop that region for quality classification.
   Else:
      a. Use the full image for quality classification.
6. Load trained Keras quality model.
7. Resize image to model input size.
8. Predict probabilities for:
      adulterated, fresh, rotten
9. Select class with maximum probability.
10. Convert class to display label and grade:
      fresh -> Fresh, Grade A
      adulterated -> Adulterant, Grade B
      rotten -> Rotten, Grade C
11. Run CLIP ViT-L/14 to estimate fruit name.
12. Draw bounding box and label on the output image.
13. Return structured JSON response.
```

### Grade Mapping

```python
GRADE_MAP = {
    "fresh": ("A", "Grade A"),
    "adulterated": ("B", "Grade B"),
    "rotten": ("C", "Grade C"),
}
```

### Missing Model Rule

If `models/fruit_quality_grader.keras` is not present, `/detect` returns an error. The app does not guess fruit quality using color thresholds or visual heuristics.

## Implementation Details

### Backend

The backend is implemented in `main.py` using FastAPI.

Main routes:

- `GET /`: returns the website
- `GET /health`: returns model and configuration status
- `POST /detect`: accepts an uploaded image and returns prediction results

The backend uses lazy loading for models. Models are loaded only when needed, which keeps startup simpler and avoids loading every model before the first request.

### Frontend

The frontend is embedded inside `main.py` as HTML, CSS, and JavaScript. It supports:

- Image upload
- Camera start and stop
- Frame capture
- Camera Robust Mode toggle
- Main result panel for grade and quality
- Secondary result fields for fruit name and model source
- Annotated image preview

The UI intentionally makes grade and quality the largest result. Fruit name is shown as supporting information.

### Camera Robust Mode

Camera Robust Mode improves difficult camera inputs. It is useful when the camera sees glare, blur, uneven brightness, or a fruit image shown on another phone screen.

Frontend behavior:

- Captures multiple frames from camera.
- Scores frames by sharpness and exposure.
- Sends the best frame to the backend.

Backend behavior:

- Attempts screen-like crop.
- Applies contrast normalization using LAB color space.
- Adjusts brightness when too dark or too bright.
- Sharpens the image before detection and grading.

### Quality Model

The deployed quality classifier is:

```text
models/fruit_quality_grader.keras
```

The model predicts three classes:

```text
adulterated
fresh
rotten
```

The current committed model is a MobileNetV2-based classifier trained on the prepared public FruitBench imagefolder data. EfficientNetV2B0 and ConvNeXtTiny training paths are also included for stronger future training.

### Fruit Detection

YOLOv8x is used to detect the fruit region. The repository does not commit the large `yolov8x.pt` file because it is a large downloaded weight file. If it is missing, Ultralytics can download it when the app first loads YOLO.

YOLO is used for:

- Finding the fruit region
- Drawing bounding boxes
- Helping the quality model focus on the fruit

YOLO is not used to decide freshness or grade.

### Fruit Naming

Fruit naming is handled by CLIP in `clip_fruit_predict.py`.

Priority:

1. CLIP `ViT-L/14`
2. CLIP `ViT-B/32`
3. VGG19 ImageNet fallback
4. MobileNetV2 ImageNet fallback

Fruit naming is helpful for display, but it is secondary to quality grading.

## Project Structure

```text
FruitQualityWeb_Rebuild/
|-- main.py
|-- clip_fruit_predict.py
|-- prepare_quality_dataset.py
|-- train_quality_backbone.py
|-- prepare_fruits360_yolo.py
|-- train_custom_yolo_fruits.py
|-- run_website.bat
|-- requirements.txt
|-- README.md
|-- models/
|   |-- fruit_quality_grader.keras
|   |-- quality_classes.json
|   `-- fruit_quality_grader.metrics.json
`-- data/
    `-- test_images/
        |-- apple.jpg
        |-- banana.jpg
        `-- pomegranate.jpg
```

Generated or downloaded folders are ignored by Git:

- `.venv/`
- `models/clip/`
- `data/quality_training/`
- `yolov8x.pt`
- logs and API response dumps

## Installation and Running

### 1. Clone the repository

```powershell
git clone https://github.com/mohammednafees1007-hub/fruit-quality-detector-for-sms.git
cd fruit-quality-detector-for-sms
```

### 2. Create virtual environment

```powershell
python -m venv .venv
```

### 3. Activate virtual environment

```powershell
.\.venv\Scripts\activate
```

### 4. Install dependencies

```powershell
pip install -r requirements.txt
```

### 5. Run the website

```powershell
.\run_website.bat
```

### 6. Open in browser

```text
http://127.0.0.1:8000/
```

## API Endpoints

### `GET /`

Returns the web interface.

### `GET /health`

Returns backend and model status.

Example fields:

```json
{
  "status": "healthy",
  "quality_model_ready": true,
  "quality_model_path": "models/fruit_quality_grader.keras",
  "quality_classes": ["adulterated", "fresh", "rotten"],
  "grading_mode": "quality_first"
}
```

### `POST /detect`

Accepts multipart form upload.

Fields:

- `file`: image file
- `robust_camera`: optional boolean

Important response fields:

```json
{
  "overall": {
    "class": "fresh",
    "label": "Fresh",
    "grade": "A",
    "grade_label": "Grade A",
    "class_conf": 0.8964,
    "probabilities": {
      "adulterated": 0.0303,
      "fresh": 0.8964,
      "rotten": 0.0733
    },
    "source": "keras_quality_model"
  },
  "fruit": {
    "name": "apple",
    "label": "Apple",
    "source": "clip_zero_shot"
  }
}
```

## Training Pipeline

### Dataset Preparation

The dataset preparation script creates a Keras-compatible imagefolder layout:

```text
data/quality_training/imagefolder/train/{adulterated,fresh,rotten}
data/quality_training/imagefolder/val/{adulterated,fresh,rotten}
data/quality_training/imagefolder/test/{adulterated,fresh,rotten}
```

Run:

```powershell
.\.venv\Scripts\python.exe prepare_quality_dataset.py
```

The split seed is deterministic:

```text
1337
```

Current label normalization:

| Source Label | Normalized Label |
| --- | --- |
| `Mature`, `Fresh`, `Good`, `Ripe` | `fresh` |
| `Rotten`, `Rot`, `Spoiled`, `Decayed` | `rotten` |
| `Pest-damage`, `Damaged`, `Formalin`, `Adulterated` | `adulterated` |
| `Unripe` | ignored |

### Current Prepared Dataset Counts

The current prepared FruitBench subset was balanced into:

| Split | Adulterated | Fresh | Rotten |
| --- | ---: | ---: | ---: |
| Train | 560 | 560 | 560 |
| Validation | 128 | 128 | 128 |
| Test | 112 | 112 | 112 |

### Training Commands

EfficientNetV2B0 target command:

```powershell
.\.venv\Scripts\python.exe train_quality_backbone.py --backbone efficientnetv2b0 --epochs 12 --fine-tune-epochs 4
```

ConvNeXtTiny comparison command:

```powershell
.\.venv\Scripts\python.exe train_quality_backbone.py --backbone convnexttiny --output models\fruit_quality_grader_convnext.keras --epochs 8 --fine-tune-epochs 0
```

CPU-friendly command used for the currently deployed model:

```powershell
.\.venv\Scripts\python.exe train_quality_backbone.py --backbone mobilenetv2 --image-size 160 --batch-size 24 --epochs 10 --fine-tune-epochs 0
```

The training script saves:

```text
models/fruit_quality_grader.keras
models/quality_classes.json
models/fruit_quality_grader.metrics.json
```

## Results and Discussion

### Current Model

The currently deployed model is:

```text
MobileNetV2, image size 160, batch size 24
```

It was selected because it performed better than the local ConvNeXtTiny run on macro-F1 in the available CPU training setup.

### Holdout Test Metrics

| Metric | Value |
| --- | ---: |
| Accuracy | 76.79% |
| Macro-F1 | 76.54% |
| Macro recall | 76.79% |

### Per-Class Results

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Adulterated | 73.39% | 71.43% | 72.40% | 112 |
| Fresh | 83.61% | 91.07% | 87.18% | 112 |
| Rotten | 72.38% | 67.86% | 70.05% | 112 |

### Confusion Matrix

Rows are true labels. Columns are predicted labels.

| True \ Predicted | Adulterated | Fresh | Rotten |
| --- | ---: | ---: | ---: |
| Adulterated | 80 | 9 | 23 |
| Fresh | 4 | 102 | 6 |
| Rotten | 25 | 11 | 76 |

### Discussion

The model performs best on fresh fruit. Rotten and adulterated samples are more difficult because visual symptoms can overlap. For example, dark marks, pest damage, bruises, and decay can look similar in an image. This explains why some rotten samples are confused with adulterated and some adulterated samples are confused with rotten.

The project reached a working quality-first system with a real trained classifier. However, the target benchmark of 80% accuracy and at least 70% recall for every class is not fully reached. Rotten recall is close but still below the target. More formalin/adulteration-specific data and longer GPU training should improve this.

## Limitations

- The deployed quality model is trained on a limited local public dataset subset.
- FruitVision is the best target dataset for formalin-mixed quality grading, but the current committed model was trained from the available FruitBench cache.
- The model can confuse rotten and adulterated classes because their visual symptoms overlap.
- Camera predictions can be affected by glare, blur, poor lighting, and phone-screen reflections.
- YOLOv8x is a generic detector, not a custom fruit detector trained specifically for this project.
- CLIP fruit naming is zero-shot and may misname visually similar fruits.
- The app is designed for local demo/presentation, not production deployment.

## Conclusion

This project implements a complete quality-first fruit grading web application. It accepts uploaded or camera images, detects the fruit region, classifies the fruit quality using a trained Keras model, maps the quality to an A/B/C grade, and displays the result in a simple website.

The main achievement is that the system no longer depends on a visual quality fallback. It uses a trained model for grading and clearly refuses prediction if the model is missing. This makes the system more honest, explainable, and suitable for presentation as a machine learning project.

## Future Enhancements

1. Train on the full FruitVision dataset
   - Use fresh, rotten, and formalin-mixed classes directly.
   - Improve the adulterated/formalin class beyond pest-damage proxy labels.

2. Train a stronger quality backbone
   - Fine-tune EfficientNetV2S or ConvNeXtTiny on GPU.
   - Use more epochs and stronger validation monitoring.
   - Deploy the model with the best macro-F1.

3. Custom YOLO fruit detector
   - Train YOLO on fruit-specific bounding boxes.
   - Add more fruit classes.
   - Improve crop quality before grading.

4. Add explainability
   - Add Grad-CAM heatmaps.
   - Show which image regions influenced quality prediction.

5. Improve camera robustness
   - Add autofocus/exposure hints.
   - Add blur rejection.
   - Capture and average multiple robust predictions.

6. Add grading history
   - Store prediction date, image, fruit name, quality, grade, and confidence.
   - Export reports as CSV or PDF.

7. Production deployment
   - Add Docker support.
   - Add cloud deployment configuration.
   - Add authentication for real SMS usage.

## References

1. FruitVision: A benchmark dataset for fresh, rotten, and formalin-mixed fruit detection. ScienceDirect Data in Brief. <https://www.sciencedirect.com/science/article/pii/S2352340925004792>
2. FruitVision original data. Mendeley Data. <https://data.mendeley.com/datasets/xkbjx8959c/2>
3. TJIET FruitBench dataset. Hugging Face. <https://huggingface.co/datasets/TJIET/FruitBench>
4. Ultralytics YOLO documentation. <https://docs.ultralytics.com/>
5. Ultralytics YOLOv8 overview. <https://www.ultralytics.com/blog/introducing-ultralytics-yolov8>
6. Radford et al. Learning Transferable Visual Models From Natural Language Supervision. arXiv:2103.00020. <https://arxiv.org/abs/2103.00020>
7. Sandler et al. MobileNetV2: Inverted Residuals and Linear Bottlenecks. arXiv:1801.04381. <https://arxiv.org/abs/1801.04381>
8. Tan and Le. EfficientNetV2: Smaller Models and Faster Training. arXiv:2104.00298. <https://arxiv.org/abs/2104.00298>
9. Liu et al. A ConvNet for the 2020s. arXiv:2201.03545. <https://arxiv.org/abs/2201.03545>
10. TensorFlow Keras Applications documentation. <https://www.tensorflow.org/api_docs/python/tf/keras/applications>
