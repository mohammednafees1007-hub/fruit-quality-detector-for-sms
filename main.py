from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import keras
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse


BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(BASE_DIR))

CLASSIFIER_PATH = Path(os.environ.get("CLASSIFIER_MODEL", BASE_DIR / "models" / "fruit_quality_grader.keras"))
QUALITY_CLASSES_PATH = Path(os.environ.get("QUALITY_CLASSES_FILE", BASE_DIR / "models" / "quality_classes.json"))
YOLO_MODEL_PATH = Path(os.environ.get("YOLO_MODEL", BASE_DIR / "yolov8x.pt"))
CLIP_MODELS = os.environ.get("CLIP_MODELS", "ViT-L/14,ViT-B/32")
CLIP_MIN_CONFIDENCE = float(os.environ.get("CLIP_MIN_CONFIDENCE", "0.04"))
YOLO_CONFIDENCE = float(os.environ.get("YOLO_CONFIDENCE", "0.12"))
YOLO_IMAGE_SIZE = int(os.environ.get("YOLO_IMAGE_SIZE", "960"))
IMG_SIZE = 224

QUALITY_CLASSES = ["adulterated", "fresh", "rotten"]
QUALITY_DISPLAY = {
    "adulterated": "Adulterant",
    "fresh": "Fresh",
    "rotten": "Rotten",
}
GRADE_MAP = {
    "fresh": ("A", "Grade A"),
    "adulterated": ("B", "Grade B"),
    "rotten": ("C", "Grade C"),
}
GRADE_DISPLAY = {
    "A": "Grade A - Fresh",
    "B": "Grade B - Adulterant / Damaged",
    "C": "Grade C - Rotten",
}
QUALITY_COLORS_BGR = {
    "fresh": (94, 197, 34),
    "rotten": (68, 68, 239),
    "adulterated": (11, 158, 245),
}

FRUIT_CLASSES = {
    "almond", "apple", "apricot", "avocado", "banana", "bean pod", "beetroot",
    "blackberry", "blueberry", "cabbage", "cactus fruit", "cantaloupe",
    "carambola", "carrot", "cashew seed", "cauliflower", "cherry", "cherimoya",
    "chestnut", "clementine", "coconut", "corn", "cucumber", "date",
    "dragon fruit", "eggplant", "fig", "ginger root", "granadilla", "grape",
    "grapefruit", "gooseberry", "guava", "hazelnut", "huckleberry", "kaki",
    "kiwi", "kohlrabi", "kumquat", "lemon", "lime", "lychee", "mandarine",
    "mango", "mangosteen", "melon", "mulberry", "nectarine", "nut", "onion",
    "orange", "papaya", "passion fruit", "peach", "peanut", "pear", "pepino",
    "pepper", "physalis", "pineapple", "pistachio", "plum", "pomegranate",
    "pomelo", "potato", "quince", "rambutan", "raspberry", "red cabbage",
    "redcurrant", "salak", "strawberry", "tamarillo", "tangelo", "tomato",
    "walnut", "watermelon", "zucchini",
}

IMAGENET_FRUIT_ALIASES = {
    "banana": "banana",
    "orange": "orange",
    "lemon": "lemon",
    "fig": "fig",
    "pineapple": "pineapple",
    "pomegranate": "pomegranate",
    "strawberry": "strawberry",
    "Granny_Smith": "apple",
    "custard_apple": "custard apple",
    "jackfruit": "jackfruit",
    "bell_pepper": "pepper",
    "cucumber": "cucumber",
    "zucchini": "zucchini",
}

_yolo = None
_quality_model = None
_quality_classes = None
_vgg19_model = None
_mobilenet_model = None

app = FastAPI(title="Fruit Quality Detector for SMS", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def display_name(name: str | None) -> str:
    if not name or name in {"unknown", "full_image"}:
        return "Unknown"
    return name.replace("_", " ").title()


def get_yolo():
    global _yolo
    if _yolo is None:
        from ultralytics import YOLO

        model_ref = str(YOLO_MODEL_PATH if YOLO_MODEL_PATH.exists() else "yolov8x.pt")
        _yolo = YOLO(model_ref)
        print(f"YOLO loaded: {model_ref}")
    return _yolo


def get_quality_model():
    global _quality_model
    if _quality_model is None:
        if not CLASSIFIER_PATH.exists():
            return None
        _quality_model = keras.models.load_model(CLASSIFIER_PATH, compile=False)
        print(f"Quality model loaded: {CLASSIFIER_PATH}")
    return _quality_model


def get_quality_classes() -> list[str]:
    global _quality_classes
    if _quality_classes is None:
        if QUALITY_CLASSES_PATH.exists():
            _quality_classes = json.loads(QUALITY_CLASSES_PATH.read_text(encoding="utf-8"))
        else:
            _quality_classes = list(QUALITY_CLASSES)
    return _quality_classes


def get_vgg19_model():
    global _vgg19_model
    if _vgg19_model is None:
        from keras.applications.vgg19 import VGG19

        _vgg19_model = VGG19(weights="imagenet")
        print("VGG19 ImageNet fallback loaded")
    return _vgg19_model


def get_mobilenet_model():
    global _mobilenet_model
    if _mobilenet_model is None:
        from keras.applications.mobilenet_v2 import MobileNetV2

        _mobilenet_model = MobileNetV2(weights="imagenet")
        print("MobileNetV2 ImageNet fallback loaded")
    return _mobilenet_model


def quality_input_size(model) -> int:
    shape = getattr(model, "input_shape", None)
    if isinstance(shape, list):
        shape = shape[0]
    if shape and len(shape) >= 3 and shape[1] and shape[2]:
        return int(shape[1])
    return IMG_SIZE


def preprocess_quality_image(image_bgr: np.ndarray, model=None) -> np.ndarray:
    image_size = quality_input_size(model) if model is not None else IMG_SIZE
    image = cv2.resize(image_bgr, (image_size, image_size), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    return image


def classify_quality(image_bgr: np.ndarray) -> dict:
    model = get_quality_model()
    if model is None:
        raise RuntimeError(
            f"Quality grading model is not trained yet. Expected model at {CLASSIFIER_PATH}. "
            "Run prepare_quality_dataset.py and train_quality_backbone.py first."
        )

    batch = np.expand_dims(preprocess_quality_image(image_bgr, model), axis=0)
    preds = model.predict(batch, verbose=0)[0]
    idx = int(np.argmax(preds))
    quality_classes = get_quality_classes()
    class_name = quality_classes[idx] if idx < len(quality_classes) else str(idx)
    grade, grade_label = GRADE_MAP.get(class_name, ("?", "Grade unavailable"))
    return {
        "class": class_name,
        "label": QUALITY_DISPLAY.get(class_name, class_name.title()),
        "grade": grade,
        "grade_label": grade_label,
        "class_conf": round(float(preds[idx]), 4),
        "probabilities": {
            quality_classes[i]: round(float(preds[i]), 4)
            for i in range(min(len(quality_classes), len(preds)))
        },
        "source": "keras_quality_model",
    }


def classify_clip_fruit(image_bgr: np.ndarray) -> dict | None:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, dir=BASE_DIR) as tmp:
            temp_path = Path(tmp.name)
        cv2.imwrite(str(temp_path), image_bgr)
        env = os.environ.copy()
        env["YOLO_CONFIG_DIR"] = str(BASE_DIR)
        env["CLIP_MODELS"] = CLIP_MODELS
        result = subprocess.run(
            [sys.executable, str(BASE_DIR / "clip_fruit_predict.py"), str(temp_path)],
            cwd=str(BASE_DIR),
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        if result.returncode != 0:
            print(f"CLIP failed: {result.stderr.strip()}")
            return None
        meta = json.loads(result.stdout.strip().splitlines()[-1])
        if float(meta.get("fruit_conf", 0.0)) < CLIP_MIN_CONFIDENCE:
            return None
        return meta
    except Exception as exc:
        print(f"CLIP failed: {exc}")
        return None
    finally:
        if temp_path:
            temp_path.unlink(missing_ok=True)


def classify_imagenet_fruit(image_bgr: np.ndarray, source: str) -> dict | None:
    try:
        if source == "vgg19_imagenet":
            from keras.applications.vgg19 import decode_predictions, preprocess_input

            model = get_vgg19_model()
        else:
            from keras.applications.mobilenet_v2 import decode_predictions, preprocess_input

            model = get_mobilenet_model()

        image = cv2.resize(image_bgr, (224, 224), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        preds = model.predict(preprocess_input(np.expand_dims(image, axis=0)), verbose=0)
        decoded = decode_predictions(preds, top=8)[0]
    except Exception as exc:
        print(f"{source} failed: {exc}")
        return None

    candidates = []
    for _, imagenet_name, score in decoded:
        clean = imagenet_name.replace(" ", "_")
        fruit = IMAGENET_FRUIT_ALIASES.get(clean)
        candidates.append({
            "imagenet_name": imagenet_name,
            "fruit_name": fruit or imagenet_name.replace("_", " "),
            "confidence": round(float(score), 4),
            "accepted": bool(fruit),
        })
        if fruit and float(score) >= 0.12:
            return {
                "fruit_name": fruit,
                "fruit_conf": round(float(score), 4),
                "fruit_source": source,
                "fruit_model": source,
                "fruit_candidates": candidates,
            }
    return None


def classify_fruit_name(image_bgr: np.ndarray) -> dict:
    for classifier in (
        classify_clip_fruit,
        lambda img: classify_imagenet_fruit(img, "vgg19_imagenet"),
        lambda img: classify_imagenet_fruit(img, "mobilenet_imagenet"),
    ):
        meta = classifier(image_bgr)
        if meta:
            return meta
    return {
        "fruit_name": "unknown",
        "fruit_conf": 0.0,
        "fruit_source": "unavailable",
        "fruit_model": None,
        "fruit_candidates": [],
    }


def order_quad_points(points: np.ndarray) -> np.ndarray:
    pts = points.reshape(4, 2).astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered


def warp_quad_region(image: np.ndarray, points: np.ndarray) -> np.ndarray | None:
    rect = order_quad_points(points)
    top_width = np.linalg.norm(rect[1] - rect[0])
    bottom_width = np.linalg.norm(rect[2] - rect[3])
    left_height = np.linalg.norm(rect[3] - rect[0])
    right_height = np.linalg.norm(rect[2] - rect[1])
    width = int(max(top_width, bottom_width))
    height = int(max(left_height, right_height))
    if width < 160 or height < 160:
        return None
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (width, height))


def crop_screen_like_region(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    image_area = float(height * width)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), dtype=np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
        area = float(cv2.contourArea(contour))
        if area < image_area * 0.10 or area > image_area * 0.98:
            continue
        approx = cv2.approxPolyDP(contour, 0.025 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            warped = warp_quad_region(image, approx)
            if warped is not None:
                return warped

        x, y, w_box, h_box = cv2.boundingRect(contour)
        if w_box >= width * 0.35 and h_box >= height * 0.35:
            pad = int(max(w_box, h_box) * 0.03)
            return image[max(0, y - pad):min(height, y + h_box + pad), max(0, x - pad):min(width, x + w_box + pad)]
    return image


def robust_camera_preprocess(image: np.ndarray) -> np.ndarray:
    work = crop_screen_like_region(image)
    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_channel = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(l_channel)
    enhanced = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
    gray_mean = float(np.mean(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)))
    if gray_mean < 95:
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.12, beta=18)
    elif gray_mean > 185:
        enhanced = cv2.convertScaleAbs(enhanced, alpha=0.92, beta=-8)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.2)
    return cv2.addWeighted(enhanced, 1.35, blurred, -0.35, 0)


def yolo_detect(image_bgr: np.ndarray) -> list[dict]:
    h, w = image_bgr.shape[:2]
    yolo = get_yolo()
    results = yolo(image_bgr, conf=YOLO_CONFIDENCE, imgsz=YOLO_IMAGE_SIZE, iou=0.45, max_det=20, verbose=False)
    detections = []
    for result in results:
        for box in result.boxes:
            coords = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, coords)
            class_id = int(box.cls[0])
            class_name = yolo.names[class_id] if hasattr(yolo, "names") else str(class_id)
            class_clean = class_name.replace("_", " ").lower()
            if class_clean not in FRUIT_CLASSES:
                continue
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "yolo_class": class_clean,
                "yolo_conf": round(float(box.conf[0]), 4),
                "fallback": False,
            })
    if not detections:
        detections.append({
            "bbox": [0, 0, w, h],
            "yolo_class": "full_image",
            "yolo_conf": 0.0,
            "fallback": True,
        })
    return detections


def select_quality_image(image_bgr: np.ndarray, detections: list[dict]) -> tuple[np.ndarray, dict | None]:
    candidates = [det for det in detections if not det.get("fallback")]
    if not candidates:
        return image_bgr, None
    h, w = image_bgr.shape[:2]
    best = max(
        candidates,
        key=lambda det: (
            max(0, det["bbox"][2] - det["bbox"][0])
            * max(0, det["bbox"][3] - det["bbox"][1])
            * max(0.05, float(det.get("yolo_conf", 0.0)))
        ),
    )
    x1, y1, x2, y2 = best["bbox"]
    pad = int(max(x2 - x1, y2 - y1) * 0.08)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return image_bgr, None
    return image_bgr[y1:y2, x1:x2], best


def scale_params(image: np.ndarray) -> tuple[float, int, int]:
    h, w = image.shape[:2]
    diag = (w * w + h * h) ** 0.5
    return max(0.45, diag / 3000), max(1, int(diag // 900)), max(6, int(diag // 220))


def draw_transparent_rect(image: np.ndarray, pt1: tuple[int, int], pt2: tuple[int, int], color: tuple[int, int, int], alpha: float = 0.45) -> None:
    overlay = image.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)


def annotate_image(image: np.ndarray, detections: list[dict], fruit: dict, quality: dict) -> np.ndarray:
    color = QUALITY_COLORS_BGR.get(quality["class"], (180, 180, 180))
    font_scale, thickness, pad = scale_params(image)
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        label = f"{quality.get('grade', '?')} - {quality['label'].upper()} {int(quality['class_conf'] * 100)}% | {display_name(fruit['fruit_name']).upper()}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        lh = th + base + pad * 2
        lx1 = x1
        ly1 = max(0, y1 - lh)
        lx2 = min(image.shape[1] - 1, lx1 + tw + pad * 2)
        ly2 = min(image.shape[0] - 1, ly1 + lh)
        draw_transparent_rect(image, (lx1, ly1), (lx2, ly2), color, alpha=0.78)
        cv2.putText(image, label, (lx1 + pad, ly2 - pad - base), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return image


def image_to_base64(image: np.ndarray) -> str:
    _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buffer).decode("utf-8")


@app.get("/", response_class=HTMLResponse)
def root():
    return """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Fruit Quality Detector for SMS</title>
    <style>
      :root { --ink:#17211b; --muted:#5d6b62; --line:#d8e2dc; --paper:#f4f8f5; --leaf:#247a45; --leaf-2:#15552d; --blue:#1f5c80; --amber:#a86612; --bad:#bd2442; --soft:#eef5f0; }
      * { box-sizing:border-box; }
      body { margin:0; min-height:100vh; font-family:Inter,system-ui,Segoe UI,sans-serif; background:linear-gradient(120deg,rgba(36,122,69,.10),transparent 38%),linear-gradient(180deg,#fbfdfb,var(--paper)); color:var(--ink); }
      main { width:min(1180px,calc(100% - 32px)); margin:0 auto; padding:28px 0; display:grid; gap:18px; }
      header { display:grid; grid-template-columns:minmax(0,1fr) auto; align-items:end; gap:16px; }
      h1 { margin:0; font-size:clamp(2rem,5vw,4rem); line-height:.98; letter-spacing:0; }
      .lead { margin:10px 0 0; color:var(--muted); max-width:760px; line-height:1.55; }
      .eyebrow { margin:0 0 8px; color:var(--leaf-2); font-weight:800; font-size:.78rem; text-transform:uppercase; }
      .status,.panel { border:1px solid var(--line); border-radius:8px; background:rgba(255,255,255,.88); }
      .status { padding:10px 12px; color:var(--muted); min-width:150px; text-align:center; font-weight:800; }
      .grid { display:grid; grid-template-columns:minmax(0,1.08fr) minmax(340px,.92fr); gap:18px; }
      .panel { box-shadow:0 18px 50px rgba(23,33,27,.10); padding:16px; }
      .preview { aspect-ratio:4/3; border-radius:8px; background:#dce8df; overflow:hidden; display:grid; place-items:center; position:relative; }
      .preview img,.preview video { width:100%; height:100%; object-fit:contain; display:none; }
      .preview video { object-fit:cover; }
      .placeholder { color:var(--muted); font-weight:700; text-align:center; padding:24px; }
      .loading { position:absolute; inset:0; background:rgba(246,250,247,.88); display:none; align-items:center; justify-content:center; padding:22px; backdrop-filter:blur(3px); }
      .loading-card { width:min(420px,100%); border:1px solid var(--line); border-radius:8px; background:white; padding:18px; box-shadow:0 14px 40px rgba(23,33,27,.12); }
      .loading-title { margin:0; font-size:1.2rem; font-weight:900; }
      .eta { margin:6px 0 12px; color:var(--muted); }
      .bar { height:10px; border-radius:999px; background:#e4ece7; overflow:hidden; }
      .bar span { display:block; height:100%; width:0%; background:linear-gradient(90deg,var(--leaf),#62a34d); transition:width .25s ease; }
      .controls { display:grid; grid-template-columns:1fr auto; gap:10px; margin-top:14px; }
      .camera-controls { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; margin-top:10px; }
      input[type=file] { width:100%; border:1px solid var(--line); border-radius:8px; background:white; padding:10px; }
      button { border:0; border-radius:8px; background:var(--leaf); color:white; min-height:44px; padding:0 18px; font-weight:800; cursor:pointer; }
      button.secondary { background:#1f5c80; } button.neutral { background:#596760; } button:disabled { opacity:.5; cursor:not-allowed; }
      .toggle { min-height:44px; border:1px solid var(--line); border-radius:8px; background:white; display:flex; align-items:center; justify-content:center; gap:8px; font-weight:800; cursor:pointer; }
      .toggle input { width:18px; height:18px; accent-color:var(--leaf); }
      h2 { margin:0; font-size:clamp(1.7rem,4vw,3rem); line-height:1; letter-spacing:0; }
      .detail { color:var(--muted); line-height:1.55; }
      .facts { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; margin:18px 0 0; }
      .fact { border:1px solid var(--line); border-radius:8px; padding:12px; background:rgba(246,250,247,.7); }
      .fact span { display:block; color:var(--muted); font-size:.78rem; font-weight:800; text-transform:uppercase; }
      .fact strong { display:block; margin-top:4px; font-size:1.12rem; overflow-wrap:anywhere; }
      .probabilities { display:grid; gap:10px; margin-top:20px; }
      .row { display:grid; grid-template-columns:minmax(0,1fr) auto; gap:12px; border-top:1px solid var(--line); padding-top:10px; color:var(--muted); }
      .row strong { color:var(--ink); } .error { color:var(--bad); font-weight:800; }
      .work { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; }
      .step { border:1px solid var(--line); border-radius:8px; background:white; padding:12px; min-height:96px; }
      .step b { display:block; margin-bottom:6px; color:var(--ink); }
      .step p { margin:0; color:var(--muted); line-height:1.42; font-size:.92rem; }
      .step.active { border-color:#86b995; background:#f1f8f3; }
      .step.done { border-color:#b8d8bf; background:#f7fbf8; }
      .process { margin-top:18px; border:1px solid var(--line); border-radius:8px; background:var(--soft); padding:12px; }
      .process-head { display:flex; justify-content:space-between; gap:12px; color:var(--muted); font-size:.9rem; font-weight:800; }
      .process-text { margin:8px 0 0; color:var(--ink); line-height:1.45; }
      .mini { color:var(--muted); font-size:.9rem; }
      @media (max-width:920px) { header,.grid { display:grid; grid-template-columns:1fr; } .work { grid-template-columns:repeat(2,minmax(0,1fr)); } }
      @media (max-width:560px) { .controls,.camera-controls,.facts,.work { grid-template-columns:1fr; } }
    </style>
  </head>
  <body>
    <main>
      <header><div><p class="eyebrow">SMS quality grading system</p><h1>Fruit Quality Detector for SMS</h1><p class="lead">Upload or capture a fruit image. The app cleans the frame when robust mode is enabled, detects fruit context, grades quality, and returns an annotated result.</p></div><div id="status" class="status">Ready</div></header>
      <section class="work" aria-label="Analysis workflow">
        <div class="step" data-step="0"><b>1. Image input</b><p>Upload a file or capture a camera frame.</p></div>
        <div class="step" data-step="1"><b>2. Frame cleanup</b><p>Robust mode improves screen or camera captures.</p></div>
        <div class="step" data-step="2"><b>3. Fruit check</b><p>Detection and fruit-name checks run on the image.</p></div>
        <div class="step" data-step="3"><b>4. Grade result</b><p>Quality is mapped to Grade A, B, or C.</p></div>
      </section>
      <section class="grid">
        <div class="panel">
          <div class="preview">
            <img id="preview" alt="Analysis result"><video id="camera" autoplay playsinline muted></video>
            <canvas id="canvas" width="1280" height="720" hidden></canvas>
            <div id="placeholder" class="placeholder">Choose a fruit image or capture from camera.</div>
            <div id="loading" class="loading"><div class="loading-card"><p id="loadingTitle" class="loading-title">Preparing image</p><p id="eta" class="eta">Estimated time: --</p><div class="bar"><span id="progressBar"></span></div><p id="loadingDetail" class="detail">Waiting for upload.</p></div></div>
          </div>
          <form id="form" class="controls"><input id="file" name="file" type="file" accept="image/*"><button id="submit" type="submit">Analyze</button></form>
          <div class="camera-controls"><button id="cameraButton" class="secondary" type="button">Start Camera</button><button id="captureButton" class="neutral" type="button" disabled>Capture Frame</button><label class="toggle"><input id="robustMode" type="checkbox" checked><span>Robust Camera</span></label></div>
        </div>
        <aside class="panel">
          <p class="eyebrow">Quality Grade</p><h2 id="title">Waiting for image</h2>
          <p id="detail" class="detail">The main result is quality and A/B/C grade. Fruit name is supporting context.</p>
          <div class="facts"><div class="fact"><span>Grade</span><strong id="gradeName">Not graded</strong></div><div class="fact"><span>Quality</span><strong id="qualityName">Waiting</strong></div><div class="fact"><span>Fruit</span><strong id="fruitName">Not detected</strong></div><div class="fact"><span>Confidence</span><strong id="confidenceName">Waiting</strong></div></div>
          <div class="process"><div class="process-head"><span>Current process</span><span id="processTime">Idle</span></div><p id="processText" class="process-text">Select an image to begin.</p></div>
          <div id="probabilities" class="probabilities"></div>
        </aside>
      </section>
    </main>
    <script>
      const form=document.querySelector("#form"),file=document.querySelector("#file"),statusEl=document.querySelector("#status"),submit=document.querySelector("#submit"),preview=document.querySelector("#preview"),camera=document.querySelector("#camera"),canvas=document.querySelector("#canvas"),cameraButton=document.querySelector("#cameraButton"),captureButton=document.querySelector("#captureButton"),robustMode=document.querySelector("#robustMode"),placeholder=document.querySelector("#placeholder"),title=document.querySelector("#title"),detail=document.querySelector("#detail"),gradeName=document.querySelector("#gradeName"),qualityName=document.querySelector("#qualityName"),fruitName=document.querySelector("#fruitName"),confidenceName=document.querySelector("#confidenceName"),probabilities=document.querySelector("#probabilities"),loading=document.querySelector("#loading"),loadingTitle=document.querySelector("#loadingTitle"),loadingDetail=document.querySelector("#loadingDetail"),eta=document.querySelector("#eta"),progressBar=document.querySelector("#progressBar"),processText=document.querySelector("#processText"),processTime=document.querySelector("#processTime"),steps=[...document.querySelectorAll(".step")];
      let stream=null,capturedBlob=null,progressTimer=null,startedAt=0;
      const pipeline=[{t:0,p:8,title:"Reading image",text:"Preparing the selected image for analysis.",step:0},{t:1200,p:24,title:"Cleaning frame",text:"Applying robust camera cleanup when enabled.",step:1},{t:2800,p:46,title:"Checking fruit",text:"Running object detection and fruit-name checks.",step:2},{t:5200,p:70,title:"Grading quality",text:"Running quality classification for Fresh, Adulterant, or Rotten.",step:3},{t:8200,p:88,title:"Drawing result",text:"Building the annotated image and result panel.",step:3}];
      function setWorkflow(index,done=false){steps.forEach((el,i)=>{el.classList.toggle("active",i===index&&!done);el.classList.toggle("done",i<index||done);});}
      function setProcess(text,time){processText.textContent=text;processTime.textContent=time;}
      function showError(message){stopProgress();title.textContent="Analysis failed";detail.innerHTML='<span class="error">'+message+"</span>";statusEl.textContent="Error";setProcess(message,"Error");loading.style.display="none";}
      function startProgress(){startedAt=Date.now();loading.style.display="flex";submit.disabled=true;probabilities.innerHTML="";progressBar.style.width="4%";setWorkflow(0);progressTimer=setInterval(()=>{const elapsed=Date.now()-startedAt;let current=pipeline[0];for(const item of pipeline){if(elapsed>=item.t)current=item;}const softProgress=Math.min(94,current.p+Math.max(0,(elapsed-current.t)/95));const remain=Math.max(2,Math.ceil((10500-elapsed)/1000));loadingTitle.textContent=current.title;loadingDetail.textContent=current.text;eta.textContent="Estimated time: about "+remain+"s";progressBar.style.width=Math.min(94,softProgress)+"%";statusEl.textContent=current.title;setWorkflow(current.step);setProcess(current.text,"About "+remain+"s left");},250);}
      function stopProgress(){if(progressTimer){clearInterval(progressTimer);progressTimer=null;}}
      function finishProgress(){stopProgress();progressBar.style.width="100%";loadingTitle.textContent="Result ready";loadingDetail.textContent="Analysis complete.";eta.textContent="Estimated time: done";setWorkflow(3,true);setProcess("Analysis complete. Result is shown above.","Done");setTimeout(()=>{loading.style.display="none";},250);}
      function stopCamera(){if(!stream)return;stream.getTracks().forEach(t=>t.stop());stream=null;camera.style.display="none";cameraButton.textContent="Start Camera";captureButton.disabled=true;}
      function showPreview(src){preview.src=src;preview.style.display="block";camera.style.display="none";placeholder.style.display="none";}
      function sleep(ms){return new Promise(r=>setTimeout(r,ms));}
      function frameScore(imageData,width,height){const data=imageData.data;let edge=0,bright=0,samples=0;const step=Math.max(2,Math.floor(Math.min(width,height)/180));for(let y=step;y<height-step;y+=step){for(let x=step;x<width-step;x+=step){const i=(y*width+x)*4,l=(y*width+x-step)*4,u=((y-step)*width+x)*4;const g=data[i]*.299+data[i+1]*.587+data[i+2]*.114,gl=data[l]*.299+data[l+1]*.587+data[l+2]*.114,gu=data[u]*.299+data[u+1]*.587+data[u+2]*.114;edge+=Math.abs(g-gl)+Math.abs(g-gu);bright+=g;samples++;}}return edge/Math.max(1,samples)-Math.abs(bright/Math.max(1,samples)-132)*.45;}
      function drawVideoFrame(){const width=camera.videoWidth||1280,height=camera.videoHeight||720;canvas.width=width;canvas.height=height;const ctx=canvas.getContext("2d",{willReadFrequently:true});ctx.drawImage(camera,0,0,width,height);const imageData=ctx.getImageData(0,0,width,height);return{width,height,imageData,score:frameScore(imageData,width,height)};}
      function enhanceFrame(frame){const out=document.createElement("canvas");out.width=frame.width;out.height=frame.height;const ctx=out.getContext("2d");ctx.putImageData(frame.imageData,0,0);if(!robustMode.checked)return out;const filtered=document.createElement("canvas");filtered.width=frame.width;filtered.height=frame.height;const fctx=filtered.getContext("2d");fctx.filter="contrast(1.18) saturate(1.08) brightness(1.04)";fctx.drawImage(out,0,0);return filtered;}
      function canvasToBlob(source){return new Promise(resolve=>source.toBlob(blob=>resolve(blob),"image/jpeg",robustMode.checked?0.95:0.92));}
      async function captureCameraBlob(){const count=robustMode.checked?8:1;let best=null;for(let i=0;i<count;i++){const frame=drawVideoFrame();if(!best||frame.score>best.score)best=frame;if(i<count-1)await sleep(90);}return await canvasToBlob(enhanceFrame(best));}
      file.addEventListener("change",()=>{capturedBlob=null;if(!file.files.length)return;stopCamera();showPreview(URL.createObjectURL(file.files[0]));statusEl.textContent="Image ready";setWorkflow(0);setProcess("Image loaded. Press Analyze to start grading.","Ready");});
      cameraButton.addEventListener("click",async()=>{if(stream){stopCamera();return;}try{stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:"environment"},audio:false});camera.srcObject=stream;await camera.play();preview.style.display="none";placeholder.style.display="none";camera.style.display="block";cameraButton.textContent="Stop Camera";captureButton.disabled=false;statusEl.textContent="Camera live";setProcess("Camera is live. Capture a frame when the fruit is clear.","Live");}catch(e){showError("Camera permission was denied or no camera is available.");}});
      captureButton.addEventListener("click",async()=>{if(!stream)return;captureButton.disabled=true;statusEl.textContent=robustMode.checked?"Capturing best frame...":"Capturing...";setProcess(robustMode.checked?"Capturing multiple frames and choosing the sharpest one.":"Capturing one frame.","Capturing");try{capturedBlob=await captureCameraBlob();file.value="";showPreview(URL.createObjectURL(capturedBlob));stopCamera();statusEl.textContent="Frame ready";setProcess("Frame captured. Press Analyze to start grading.","Ready");}catch(e){showError("Could not capture a clean camera frame.");captureButton.disabled=false;}});
      form.addEventListener("submit",async(event)=>{event.preventDefault();if(!file.files.length&&!capturedBlob){showError("Choose an image or capture a camera frame first.");return;}startProgress();const data=new FormData();data.append("file",capturedBlob||file.files[0],capturedBlob?"camera-robust-frame.jpg":file.files[0].name);if(capturedBlob&&robustMode.checked)data.append("robust_camera","true");try{const response=await fetch("/detect",{method:"POST",body:data});const result=await response.json();if(!response.ok)throw new Error(result.detail||"Backend could not grade this image.");const overall=result.overall||{},fruit=result.fruit||{};finishProgress();title.textContent=(overall.grade||"?")+" - "+(overall.label||"Quality");gradeName.textContent=overall.grade_label||overall.grade||"Unknown";qualityName.textContent=overall.label||"Unknown";fruitName.textContent=fruit.label||"Unknown";confidenceName.textContent=Math.round((overall.class_conf||0)*100)+"%";detail.textContent="Fruit: "+(fruit.label||"Unknown")+" ("+Math.round((fruit.confidence||0)*100)+"%). Quality confidence: "+Math.round((overall.class_conf||0)*100)+"%.";preview.src="data:image/jpeg;base64,"+result.annotated_image;preview.style.display="block";placeholder.style.display="none";statusEl.textContent="Done";probabilities.innerHTML=Object.entries(overall.probabilities||{}).map(([k,v])=>'<div class="row"><strong>'+k+'</strong><span>'+Math.round(v*100)+'%</span></div>').join("");}catch(error){showError(error.message||"Unknown error");}finally{submit.disabled=false;}});
    </script>
  </body>
</html>
"""


@app.get("/health")
def health():
    clip_files = sorted(path.name for path in (BASE_DIR / "models" / "clip").glob("*") if path.is_file())
    quality_classes = get_quality_classes() if QUALITY_CLASSES_PATH.exists() else list(QUALITY_CLASSES)
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "base_dir": str(BASE_DIR),
        "yolo_model": str(YOLO_MODEL_PATH),
        "yolo_model_exists": YOLO_MODEL_PATH.exists(),
        "quality_model": str(CLASSIFIER_PATH),
        "quality_model_exists": CLASSIFIER_PATH.exists(),
        "quality_model_ready": CLASSIFIER_PATH.exists() and QUALITY_CLASSES_PATH.exists(),
        "quality_model_path": str(CLASSIFIER_PATH),
        "quality_classes": quality_classes,
        "grading_mode": "quality_first",
        "clip_model_priority": [item.strip() for item in CLIP_MODELS.split(",") if item.strip()],
        "clip_cached_files": clip_files,
        "vgg19_fallback": True,
        "mobilenet_fallback": True,
    }


@app.post("/detect")
async def detect(
    request: Request,
    file: UploadFile = File(...),
    robust_camera: bool = Form(False),
):
    query_robust = request.query_params.get("robust_camera")
    if query_robust is not None:
        robust_camera = query_robust.lower() in {"1", "true", "yes", "on"}

    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    raw_image = image.copy()
    input_h, input_w = image.shape[:2]
    robust_requested = bool(robust_camera) or (file.filename or "").lower().startswith("camera-robust")
    if robust_requested:
        image = robust_camera_preprocess(image)

    detections = yolo_detect(image)
    fruit_meta = classify_fruit_name(raw_image if robust_requested else image)
    quality_image, quality_detection = select_quality_image(image, detections)
    try:
        quality = classify_quality(quality_image)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    final_detections = []
    for det in detections:
        final_detections.append({
            **det,
            "fruit_name": fruit_meta["fruit_name"],
            "fruit_label": display_name(fruit_meta["fruit_name"]),
            "fruit_conf": fruit_meta["fruit_conf"],
            "fruit_source": fruit_meta["fruit_source"],
            "class": quality["class"],
            "label": quality["label"],
            "class_conf": quality["class_conf"],
        })

    annotated = annotate_image(image.copy(), final_detections, fruit_meta, quality)
    h, w = image.shape[:2]
    return {
        "annotated_image": image_to_base64(annotated),
        "detections": final_detections,
        "overall": quality,
        "fruit": {
            "name": fruit_meta["fruit_name"],
            "label": display_name(fruit_meta["fruit_name"]),
            "confidence": fruit_meta["fruit_conf"],
            "source": fruit_meta["fruit_source"],
            "model": fruit_meta.get("fruit_model"),
            "candidates": fruit_meta.get("fruit_candidates", []),
        },
        "preprocessing": {
            "robust_camera": robust_requested,
            "input_size": {"width": int(input_w), "height": int(input_h)},
            "processed_size": {"width": int(w), "height": int(h)},
            "quality_crop": quality_detection["bbox"] if quality_detection else None,
        },
        "total": len(final_detections),
        "used_fallback": any(det.get("fallback", False) for det in detections),
    }
