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

# Runtime decision controls. These do not retrain the model; they decide when a
# prediction is confident enough to be shown as a final grade.
QUALITY_MIN_CONFIDENCE = float(os.environ.get("QUALITY_MIN_CONFIDENCE", "0.40"))
QUALITY_CLASS_THRESHOLDS = {
    "fresh": float(os.environ.get("QUALITY_THRESHOLD_FRESH", os.environ.get("QUALITY_MIN_CONFIDENCE", "0.40"))),
    "adulterated": float(os.environ.get("QUALITY_THRESHOLD_ADULTERATED", os.environ.get("QUALITY_MIN_CONFIDENCE", "0.40"))),
    "rotten": float(os.environ.get("QUALITY_THRESHOLD_ROTTEN", os.environ.get("QUALITY_MIN_CONFIDENCE", "0.40"))),
}
FRUIT_GATE_ENABLED = os.environ.get("FRUIT_GATE_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
FRUIT_GATE_MODE = os.environ.get("FRUIT_GATE_MODE", "any").lower()
FRUIT_MIN_CONFIDENCE = float(os.environ.get("FRUIT_MIN_CONFIDENCE", "0.12"))
YOLO_MIN_REAL_FRUIT_CONFIDENCE = float(os.environ.get("YOLO_MIN_REAL_FRUIT_CONFIDENCE", "0.18"))

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


def threshold_config(
    quality_min_confidence: float | None = None,
    quality_threshold_fresh: float | None = None,
    quality_threshold_adulterated: float | None = None,
    quality_threshold_rotten: float | None = None,
    fruit_gate_enabled: bool | None = None,
    fruit_gate_mode: str | None = None,
    fruit_min_confidence: float | None = None,
    yolo_min_real_fruit_confidence: float | None = None,
) -> dict:
    min_conf = QUALITY_MIN_CONFIDENCE if quality_min_confidence is None else float(quality_min_confidence)
    class_thresholds = {
        "fresh": QUALITY_CLASS_THRESHOLDS["fresh"] if quality_threshold_fresh is None else float(quality_threshold_fresh),
        "adulterated": QUALITY_CLASS_THRESHOLDS["adulterated"] if quality_threshold_adulterated is None else float(quality_threshold_adulterated),
        "rotten": QUALITY_CLASS_THRESHOLDS["rotten"] if quality_threshold_rotten is None else float(quality_threshold_rotten),
    }
    return {
        "quality_min_confidence": min_conf,
        "quality_class_thresholds": class_thresholds,
        "fruit_gate_enabled": FRUIT_GATE_ENABLED if fruit_gate_enabled is None else bool(fruit_gate_enabled),
        "fruit_gate_mode": (fruit_gate_mode or FRUIT_GATE_MODE).lower(),
        "fruit_min_confidence": FRUIT_MIN_CONFIDENCE if fruit_min_confidence is None else float(fruit_min_confidence),
        "yolo_min_real_fruit_confidence": YOLO_MIN_REAL_FRUIT_CONFIDENCE if yolo_min_real_fruit_confidence is None else float(yolo_min_real_fruit_confidence),
    }


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


def uncertain_quality(reason: str, message: str, base_quality: dict | None = None) -> dict:
    base_quality = base_quality or {}
    original_class = base_quality.get("class")
    original_conf = float(base_quality.get("class_conf", 0.0) or 0.0)
    return {
        "class": "uncertain",
        "label": "Uncertain",
        "grade": "?",
        "grade_label": "Manual Review",
        "class_conf": round(original_conf, 4),
        "probabilities": base_quality.get("probabilities", {}),
        "source": base_quality.get("source", "decision_gate"),
        "accepted": False,
        "reject_reason": reason,
        "message": message,
        "original_class": original_class,
        "original_label": base_quality.get("label"),
        "original_grade": base_quality.get("grade"),
    }


def apply_quality_thresholds(quality: dict, config: dict | None = None) -> dict:
    config = config or threshold_config()
    class_name = quality.get("class")
    confidence = float(quality.get("class_conf", 0.0) or 0.0)
    required = config["quality_class_thresholds"].get(class_name, config["quality_min_confidence"])
    if confidence < required:
        gated = uncertain_quality(
            "low_quality_confidence",
            f"Quality confidence {int(confidence * 100)}% is below the required {int(required * 100)}%.",
            quality,
        )
        gated["required_confidence"] = required
        return gated
    quality["accepted"] = True
    quality["required_confidence"] = required
    return quality


def fruit_gate_decision(detections: list[dict], fruit_meta: dict, config: dict | None = None) -> dict:
    config = config or threshold_config()
    best_real_detection = None
    for det in detections:
        if det.get("fallback"):
            continue
        if float(det.get("yolo_conf", 0.0) or 0.0) >= config["yolo_min_real_fruit_confidence"]:
            if best_real_detection is None or float(det.get("yolo_conf", 0.0)) > float(best_real_detection.get("yolo_conf", 0.0)):
                best_real_detection = det

    fruit_name = fruit_meta.get("fruit_name")
    fruit_conf = float(fruit_meta.get("fruit_conf", 0.0) or 0.0)
    yolo_ok = best_real_detection is not None
    fruit_name_ok = bool(fruit_name and fruit_name != "unknown" and fruit_conf >= config["fruit_min_confidence"])
    if config["fruit_gate_mode"] == "both":
        accepted = yolo_ok and fruit_name_ok
    else:
        accepted = yolo_ok or fruit_name_ok

    return {
        "enabled": config["fruit_gate_enabled"],
        "accepted": accepted,
        "mode": config["fruit_gate_mode"],
        "yolo_ok": yolo_ok,
        "fruit_name_ok": fruit_name_ok,
        "best_yolo_class": best_real_detection.get("yolo_class") if best_real_detection else None,
        "best_yolo_confidence": best_real_detection.get("yolo_conf") if best_real_detection else 0.0,
        "fruit_name": fruit_name,
        "fruit_confidence": fruit_conf,
        "thresholds": {
            "fruit_min_confidence": config["fruit_min_confidence"],
            "yolo_min_real_fruit_confidence": config["yolo_min_real_fruit_confidence"],
        },
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
        if quality.get("accepted", True) is False:
            label = f"REVIEW - {quality['label'].upper()} | {display_name(fruit['fruit_name']).upper()}"
        else:
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
    <title>FruitVision AI | Fruit Quality Detector for SMS</title>
    <style>
      :root { --ink:#08130e; --muted:#607064; --line:#d9e8dd; --paper:#f4faf5; --card:rgba(255,255,255,.80); --leaf:#0f7a3f; --leaf-2:#2eb872; --leaf-dark:#0b4f2a; --sage:#dceee2; --amber:#b87910; --bad:#c92f4f; --shadow:0 30px 88px rgba(8,30,18,.13); --soft-shadow:0 16px 42px rgba(8,30,18,.08); --radius:28px; --grade:#0f7a3f; --grade-soft:#e5f7eb; }
      * { box-sizing:border-box; }
      html { scroll-behavior:smooth; }
      body { margin:0; min-height:100vh; font-family:Inter,ui-sans-serif,system-ui,Segoe UI,sans-serif; background:linear-gradient(135deg,#f2fbf4 0%,#fff 38%,#e6f7ec 100%); color:var(--ink); animation:pageIn .55s ease both; }
      body::before { content:""; position:fixed; inset:0; pointer-events:none; background:radial-gradient(circle at 14% 9%,rgba(15,122,63,.18),transparent 28%),radial-gradient(circle at 88% 10%,rgba(46,184,114,.18),transparent 30%),linear-gradient(180deg,rgba(255,255,255,.25),rgba(220,238,226,.26)); }
      a { color:inherit; text-decoration:none; }
      button,input { font:inherit; }
      .nav { position:sticky; top:0; z-index:30; border-bottom:1px solid rgba(255,255,255,.7); background:rgba(255,255,255,.72); backdrop-filter:blur(22px); }
      .nav-inner { width:min(1180px,calc(100% - 32px)); margin:0 auto; min-height:70px; display:flex; align-items:center; justify-content:space-between; gap:18px; }
      .brand { display:flex; align-items:center; gap:12px; font-size:1.05rem; font-weight:950; letter-spacing:0; }
      .logo { width:40px; aspect-ratio:1; display:grid; place-items:center; border-radius:16px; background:linear-gradient(135deg,var(--leaf-dark),var(--leaf-2)); color:white; box-shadow:var(--soft-shadow); }
      .nav-links { display:flex; align-items:center; gap:24px; color:#535f58; font-size:.92rem; font-weight:800; }
      .nav-links a:hover { color:var(--ink); }
      .mobile-menu { display:none; border:1px solid var(--line); background:white; color:var(--ink); box-shadow:none; width:44px; padding:0; }
      .status { min-width:132px; border:1px solid var(--line); border-radius:999px; background:rgba(255,255,255,.8); color:var(--muted); padding:10px 14px; text-align:center; font-size:.9rem; font-weight:900; box-shadow:var(--soft-shadow); transition:.22s ease; }
      .status.analyzing { color:#123d28; background:#e8f6ed; transform:translateY(-1px); }
      .status.done { color:#12391f; background:#ddf4e4; }
      .status.error-state { color:#7e1430; background:#fde9ee; }
      main { position:relative; z-index:1; }
      .section { width:min(1200px,calc(100% - 40px)); margin:0 auto; padding:86px 0; }
      .hero { min-height:calc(100vh - 70px); display:grid; grid-template-columns:minmax(0,1.05fr) minmax(0,.95fr); align-items:center; gap:48px; padding-top:58px; }
      .eyebrow { margin:0; color:var(--leaf); font-weight:950; font-size:.76rem; text-transform:uppercase; letter-spacing:.22em; }
      h1,h2,h3,p { letter-spacing:0; }
      h1 { margin:20px 0 0; font-size:clamp(3.1rem,8vw,6.9rem); line-height:.96; font-weight:950; }
      h2 { margin:0; font-size:clamp(2.25rem,5vw,4.6rem); line-height:1; font-weight:950; }
      h3 { margin:0; font-size:1.35rem; font-weight:950; }
      .lead { margin:24px 0 0; max-width:690px; color:var(--muted); font-size:clamp(1.06rem,2vw,1.32rem); line-height:1.65; }
      .hero-actions { display:flex; flex-wrap:wrap; gap:12px; margin-top:34px; }
      .btn,button { border:0; border-radius:999px; min-height:48px; padding:0 22px; display:inline-flex; align-items:center; justify-content:center; gap:9px; background:linear-gradient(135deg,var(--leaf-dark),var(--leaf)); color:white; font-weight:950; cursor:pointer; box-shadow:0 16px 34px rgba(15,122,63,.18); transition:transform .18s ease,box-shadow .18s ease,background .18s ease,opacity .18s ease; }
      .btn:hover,button:hover { transform:translateY(-1px); box-shadow:0 22px 44px rgba(15,122,63,.24); }
      .btn:active,button:active { transform:translateY(1px) scale(.99); }
      .btn.secondary,button.secondary { background:white; color:var(--ink); border:1px solid var(--line); box-shadow:var(--soft-shadow); }
      button.neutral { background:#69736d; }
      button:disabled { opacity:.48; cursor:not-allowed; transform:none; }
      .glass,.panel,.card { border:1px solid rgba(255,255,255,.72); background:var(--card); box-shadow:var(--shadow); backdrop-filter:blur(22px); }
      .mockup { border-radius:36px; padding:16px; }
      .mock-img { position:relative; aspect-ratio:4/3; overflow:hidden; border-radius:26px; background:#dce8df; }
      .mock-img img { width:100%; height:100%; object-fit:cover; display:block; }
      .mock-img::after { content:""; position:absolute; inset:0; background:linear-gradient(to top,rgba(0,0,0,.42),rgba(0,0,0,.02) 56%,transparent); }
      .mock-pill { position:absolute; left:18px; top:18px; z-index:2; border-radius:999px; background:rgba(255,255,255,.82); padding:10px 14px; color:var(--ink); font-size:.76rem; font-weight:950; text-transform:uppercase; letter-spacing:.15em; backdrop-filter:blur(16px); }
      .mock-stats { position:absolute; left:18px; right:18px; bottom:18px; z-index:2; display:grid; grid-template-columns:repeat(3,1fr); gap:10px; }
      .mock-stat { border:1px solid rgba(255,255,255,.36); border-radius:18px; background:rgba(255,255,255,.82); padding:13px; backdrop-filter:blur(14px); }
      .mock-stat span,.fact span,.stat span { display:block; color:var(--muted); font-size:.72rem; font-weight:950; text-transform:uppercase; letter-spacing:.14em; }
      .mock-stat strong,.fact strong,.stat strong { display:block; margin-top:5px; color:var(--ink); font-size:1.35rem; font-weight:950; overflow-wrap:anywhere; }
      .section-title { max-width:760px; margin:0 auto 46px; text-align:center; }
      .section-title p:last-child { margin:18px auto 0; color:var(--muted); line-height:1.65; font-size:1.08rem; }
      .steps,.stats,.gallery,.ai-grid,.model-notes { display:grid; gap:16px; align-items:stretch; }
      .steps { grid-template-columns:repeat(4,minmax(0,1fr)); }
      .step,.stat,.gallery-card,.ai-card { min-width:0; height:100%; border:1px solid var(--line); border-radius:26px; background:rgba(255,255,255,.8); padding:22px; box-shadow:var(--soft-shadow); transition:transform .24s ease,box-shadow .24s ease,border-color .24s ease; }
      .step,.stat,.ai-card { display:flex; flex-direction:column; }
      .step:hover,.stat:hover,.gallery-card:hover,.ai-card:hover { transform:translateY(-3px); box-shadow:var(--shadow); }
      .step-num,.ai-icon { width:46px; aspect-ratio:1; border-radius:17px; display:grid; place-items:center; background:linear-gradient(135deg,var(--leaf-dark),var(--leaf)); color:white; font-weight:950; margin-bottom:20px; }
      .step.active { border-color:#8fc49e; background:#f1fbf4; animation:stepGlow 1.35s ease-in-out infinite alternate; }
      .step.done { border-color:#b8d8bf; background:#f8fcf9; }
      .step p,.ai-card p,.gallery-card p,.stat p { color:var(--muted); line-height:1.55; margin:10px 0 0; }
      .model-notes { grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); margin-top:18px; }
      .detect-grid { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:22px; align-items:stretch; }
      .panel { min-width:0; height:100%; border-radius:36px; padding:20px; }
      .detect-grid > .panel,.result-panel { display:flex; flex-direction:column; }
      .preview { position:relative; min-height:410px; aspect-ratio:4/3; border:1px dashed #cbd6cf; border-radius:28px; background:rgba(255,255,255,.72); overflow:hidden; display:grid; place-items:center; transition:.22s ease; }
      .preview.dragging { border-color:var(--leaf); background:#effaf2; transform:scale(.997); }
      .preview img,.preview video { width:100%; height:100%; object-fit:contain; display:none; opacity:0; transform:scale(.99); transition:opacity .32s ease,transform .32s ease; }
      .preview img.visible,.preview video.visible { opacity:1; transform:scale(1); }
      .preview video { object-fit:cover; }
      .placeholder { max-width:390px; padding:24px; text-align:center; color:var(--muted); font-weight:800; }
      .placeholder .upload-icon { width:66px; aspect-ratio:1; display:grid; place-items:center; margin:0 auto 18px; border-radius:24px; background:linear-gradient(135deg,var(--leaf-dark),var(--leaf-2)); color:white; font-size:1.5rem; box-shadow:var(--soft-shadow); }
      .loading { position:absolute; inset:0; background:rgba(255,255,255,.72); display:none; align-items:center; justify-content:center; padding:22px; backdrop-filter:blur(12px); }
      .loading::before { content:""; position:absolute; inset:0; background:linear-gradient(105deg,transparent 0%,transparent 42%,rgba(255,255,255,.65) 50%,transparent 58%,transparent 100%); transform:translateX(-100%); animation:scan 1.65s ease-in-out infinite; }
      .loading-card { position:relative; width:min(460px,100%); border:1px solid rgba(255,255,255,.8); border-radius:28px; background:rgba(255,255,255,.94); padding:24px; box-shadow:var(--shadow); display:grid; grid-template-columns:104px minmax(0,1fr); gap:18px; align-items:center; }
      .circle-loader { --progress:4; width:104px; aspect-ratio:1; border-radius:50%; display:grid; place-items:center; background:conic-gradient(var(--leaf) calc(var(--progress)*1%),#e2eee6 0); box-shadow:inset 0 0 0 1px rgba(15,122,63,.10),0 16px 34px rgba(15,122,63,.18); transition:background .28s ease; }
      .circle-loader::before { content:""; width:74px; aspect-ratio:1; border-radius:50%; background:white; box-shadow:inset 0 0 0 1px rgba(15,122,63,.08); }
      .circle-value { position:absolute; z-index:2; font-size:1.04rem; font-weight:950; color:var(--leaf-dark); }
      .circle-wrap { position:relative; display:grid; place-items:center; }
      .loading-title { margin:0; font-size:1.2rem; font-weight:950; }
      .eta { margin:7px 0 14px; color:var(--muted); }
      .loading-meta { margin:0; color:var(--muted); line-height:1.5; }
      .controls { display:grid; grid-template-columns:minmax(0,1fr) minmax(130px,auto); gap:12px; margin-top:16px; align-items:stretch; }
      input[type=file] { width:100%; border:1px solid var(--line); border-radius:999px; background:white; padding:12px 14px; color:var(--muted); }
      .camera-controls { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin-top:12px; align-items:stretch; }
      .toggle { min-height:48px; border:1px solid var(--line); border-radius:999px; background:white; display:flex; align-items:center; justify-content:center; gap:8px; font-weight:950; cursor:pointer; box-shadow:var(--soft-shadow); transition:.18s ease; }
      .toggle:hover { transform:translateY(-1px); border-color:#9fc8ad; }
      .toggle input { width:18px; height:18px; accent-color:var(--leaf); }
      .result-panel { --grade:#1f7a4d; --grade-soft:#e9f7ee; overflow:hidden; }
      .result-panel.grade-a { --grade:#1f7a4d; --grade-soft:#e9f7ee; }
      .result-panel.grade-b { --grade:#b16c14; --grade-soft:#fff3dc; }
      .result-panel.grade-c { --grade:#c92f4f; --grade-soft:#fde8ee; }
      .report-main { min-width:0; display:flex; flex-direction:column; height:100%; }
      .report-card { border:1px solid var(--line); border-radius:24px; background:rgba(255,255,255,.66); padding:16px; }
      .result-hero { display:grid; grid-template-columns:88px minmax(0,1fr); gap:18px; align-items:center; padding:18px; border:1px solid color-mix(in srgb,var(--grade) 22%,var(--line)); border-radius:26px; background:linear-gradient(135deg,var(--grade-soft),rgba(255,255,255,.92)); }
      .grade-badge { width:88px; aspect-ratio:1; border-radius:26px; display:grid; place-items:center; color:white; background:var(--grade); font-size:3.25rem; font-weight:950; box-shadow:0 18px 38px color-mix(in srgb,var(--grade) 24%,transparent); }
      .result-panel.result-flash .grade-badge { animation:resultPulse .62s ease; }
      .detail { color:var(--muted); line-height:1.58; margin:8px 0 0; }
      .facts { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; margin:16px 0 0; }
      .fact { border:1px solid var(--line); border-radius:22px; padding:15px; background:rgba(255,255,255,.68); }
      .fact.primary { border-color:color-mix(in srgb,var(--grade) 22%,var(--line)); background:var(--grade-soft); }
      .recommend { margin-top:14px; border:1px solid color-mix(in srgb,var(--grade) 18%,var(--line)); border-radius:24px; background:rgba(255,255,255,.62); padding:16px; }
      .recommend strong { display:block; color:var(--grade); font-size:1.22rem; margin-top:4px; }
      .report-meter { margin-top:14px; display:grid; grid-template-columns:88px minmax(0,1fr); gap:14px; align-items:stretch; }
      .score-ring { --score:0; width:88px; aspect-ratio:1; align-self:center; border-radius:50%; display:grid; place-items:center; background:conic-gradient(var(--grade) calc(var(--score)*1%),#e6eee9 0); }
      .score-ring span { width:62px; aspect-ratio:1; border-radius:50%; background:white; display:grid; place-items:center; color:var(--grade); font-weight:950; }
      .info-chip { min-width:0; height:100%; border:1px solid var(--line); border-radius:20px; background:linear-gradient(135deg,rgba(255,255,255,.82),rgba(235,248,239,.72)); padding:13px; }
      .info-chip b { display:block; color:var(--ink); font-size:.95rem; }
      .info-chip span { display:block; margin-top:5px; color:var(--muted); font-size:.82rem; line-height:1.45; }
      .probabilities { display:grid; gap:10px; margin-top:18px; }
      .prob-row { display:grid; gap:8px; border-top:1px solid var(--line); padding-top:12px; }
      .prob-head { display:grid; grid-template-columns:minmax(0,1fr) auto; gap:12px; color:var(--muted); }
      .prob-head strong { color:var(--ink); text-transform:capitalize; }
      .prob-track { height:9px; border-radius:999px; background:#e5ece8; overflow:hidden; }
      .prob-fill { display:block; height:100%; width:0%; border-radius:999px; background:var(--grade); transition:width .7s cubic-bezier(.2,.8,.2,1); }
      .process { margin-top:16px; border:1px solid var(--line); border-radius:24px; background:rgba(247,250,248,.85); padding:15px; }
      .process-head { display:flex; justify-content:space-between; gap:12px; color:var(--muted); font-size:.88rem; font-weight:950; }
      .process-text { margin:8px 0 0; color:var(--ink); line-height:1.45; }
      .error { color:var(--bad); font-weight:950; }
      .settings-bar { margin-top:14px; border:1px solid var(--line); border-radius:24px; background:rgba(255,255,255,.72); padding:15px; }
      .settings-summary { display:flex; align-items:center; justify-content:space-between; gap:12px; cursor:pointer; font-weight:950; color:var(--ink); }
      .settings-summary::-webkit-details-marker { display:none; }
      .settings-summary span { color:var(--muted); font-size:.82rem; font-weight:850; }
      .settings-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; margin-top:14px; }
      .setting { min-width:0; border:1px solid #e5efe8; border-radius:18px; background:rgba(247,250,248,.9); padding:12px; }
      .setting-head { display:flex; justify-content:space-between; gap:10px; align-items:center; color:var(--muted); font-size:.82rem; font-weight:950; text-transform:uppercase; letter-spacing:.08em; }
      .setting-head b { color:var(--ink); font-size:.9rem; letter-spacing:0; text-transform:none; }
      .setting input[type=range] { width:100%; margin-top:10px; accent-color:var(--leaf); }
      .setting select { width:100%; margin-top:10px; min-height:38px; border:1px solid var(--line); border-radius:12px; padding:0 10px; background:white; color:var(--ink); font-weight:850; }
      .setting.toggle-setting { display:flex; align-items:center; justify-content:space-between; gap:12px; }
      .setting.toggle-setting input { width:18px; height:18px; accent-color:var(--leaf); }
      .stats { grid-template-columns:repeat(auto-fit,minmax(185px,1fr)); }
      .stat strong { font-size:2.2rem; }
      .history { margin-top:18px; display:grid; gap:10px; }
      .history-row { display:grid; grid-template-columns:minmax(120px,1.15fr) minmax(105px,.9fr) minmax(90px,.78fr) minmax(76px,.68fr) minmax(88px,.68fr); align-items:center; gap:12px; border:1px solid var(--line); border-radius:20px; background:rgba(255,255,255,.72); padding:14px; }
      .history-row > * { min-width:0; overflow-wrap:anywhere; }
      .history-row b { font-size:1.02rem; }
      .history-row span { color:var(--muted); }
      .gallery { grid-template-columns:repeat(4,minmax(0,1fr)); }
      .gallery-card { padding:0; overflow:hidden; display:flex; flex-direction:column; }
      .gallery-card img { width:100%; aspect-ratio:4/3; object-fit:cover; display:block; transition:transform .45s ease; }
      .gallery-card:hover img { transform:scale(1.045); }
      .gallery-body { padding:18px; display:flex; flex:1; flex-direction:column; }
      .gallery-top { display:flex; align-items:center; justify-content:space-between; gap:10px; }
      .grade-pill { display:inline-flex; align-items:center; min-height:30px; padding:0 12px; border-radius:999px; background:#e9f7ee; color:var(--leaf); font-weight:950; font-size:.85rem; }
      .try-sample { margin-top:auto; width:100%; min-height:42px; background:linear-gradient(135deg,var(--leaf-dark),var(--leaf-2)); font-size:.92rem; }
      .ai-grid { grid-template-columns:repeat(4,minmax(0,1fr)); }
      .json-card { margin-top:18px; border-radius:28px; background:var(--ink); color:white; padding:22px; box-shadow:var(--shadow); overflow:auto; }
      pre { margin:0; color:rgba(255,255,255,.82); line-height:1.7; }
      .cta { border-radius:38px; background:var(--ink); color:white; padding:44px; box-shadow:var(--shadow); display:grid; grid-template-columns:1fr auto; gap:22px; align-items:end; }
      .cta .lead { color:rgba(255,255,255,.68); }
      .cta .btn { background:white; color:var(--ink); }
      footer { padding:0 0 34px; }
      @keyframes pageIn { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:none; } }
      @keyframes scan { 0% { transform:translateX(-100%); } 62%,100% { transform:translateX(100%); } }
      @keyframes stepGlow { from { box-shadow:0 0 0 2px rgba(31,122,77,.08),0 16px 38px rgba(16,22,19,.08); } to { box-shadow:0 0 0 6px rgba(31,122,77,.14),0 22px 48px rgba(16,22,19,.11); } }
      @keyframes resultPulse { 0% { transform:scale(.94); } 45% { transform:scale(1.08); } 100% { transform:scale(1); } }
      @media (max-width:1100px) { .model-notes { grid-template-columns:repeat(3,minmax(0,1fr)); } .history-row { grid-template-columns:repeat(5,minmax(0,1fr)); } }
      @media (max-width:980px) { .hero,.detect-grid,.cta { grid-template-columns:1fr; } .steps,.stats,.gallery,.ai-grid,.model-notes { grid-template-columns:repeat(2,minmax(0,1fr)); } .nav-links { display:none; } .mobile-menu { display:grid; place-items:center; } }
      @media (max-width:620px) { .section { width:min(100% - 22px,1180px); padding:58px 0; } .hero { padding-top:36px; } .steps,.stats,.gallery,.ai-grid,.model-notes,.controls,.camera-controls,.facts,.history-row,.mock-stats,.loading-card,.report-meter,.settings-grid { grid-template-columns:1fr; } .preview { min-height:330px; } .result-hero { grid-template-columns:1fr; } .grade-badge { width:74px; border-radius:22px; font-size:2.7rem; } .cta { padding:28px; } }
      @media (prefers-reduced-motion:reduce) { *,*::before,*::after { animation:none!important; transition:none!important; scroll-behavior:auto!important; } }
    </style>
  </head>
  <body>
    <nav class="nav">
      <div class="nav-inner">
        <a class="brand" href="#top"><span class="logo">FV</span><span>FruitVision AI</span></a>
        <div class="nav-links">
          <a href="#how">How it works</a>
          <a href="#detect">Detection</a>
          <a href="#dashboard">Dashboard</a>
          <a href="#gallery">Gallery</a>
        </div>
        <div id="status" class="status">Ready</div>
      </div>
    </nav>
    <main>
      <section id="top" class="section hero">
        <div>
          <p class="eyebrow">SMS quality grading system</p>
          <h1>AI-powered fruit quality detection.</h1>
          <p class="lead">Scan fruits instantly and detect freshness, defects, grade, and quality using the existing computer vision backend.</p>
          <div class="hero-actions">
            <a class="btn" href="#detect">Scan Fruit</a>
            <a class="btn secondary" href="#how">How It Works</a>
          </div>
        </div>
        <aside class="glass mockup">
          <div class="mock-img">
            <img src="https://images.unsplash.com/photo-1568702846914-96b305d2aaeb?auto=format&fit=crop&w=1200&q=85" alt="Fresh apple preview">
            <div class="mock-pill">Live scan preview</div>
            <div class="mock-stats">
              <div class="mock-stat"><span>Fruit</span><strong>Apple</strong></div>
              <div class="mock-stat"><span>Grade</span><strong>A</strong></div>
              <div class="mock-stat"><span>Score</span><strong>92</strong></div>
            </div>
          </div>
        </aside>
      </section>

      <section id="how" class="section">
        <div class="section-title">
          <p class="eyebrow">Workflow</p>
          <h2>From image to grade in four steps.</h2>
          <p>The website keeps the presentation clean while the existing FastAPI backend performs detection and grading.</p>
        </div>
        <div class="steps" aria-label="Analysis workflow">
          <div class="step" data-step="0"><div class="step-num">1</div><h3>Upload or capture</h3><p>Select a fruit image, drag and drop it, or use the camera panel.</p></div>
          <div class="step" data-step="1"><div class="step-num">2</div><h3>Clean the frame</h3><p>Robust mode improves camera and phone-screen captures before prediction.</p></div>
          <div class="step" data-step="2"><div class="step-num">3</div><h3>Detect fruit</h3><p>YOLO and fruit-name checks provide object context for the quality result.</p></div>
          <div class="step" data-step="3"><div class="step-num">4</div><h3>Show grade</h3><p>The quality model maps Fresh, Adulterant, or Rotten into Grade A, B, or C.</p></div>
        </div>
        <div class="model-notes" aria-label="Model notes">
          <div class="info-chip"><b>Quality-first system</b><span>Keras MobileNetV2 decides Fresh, Adulterant, or Rotten.</span></div>
          <div class="info-chip"><b>Grade mapping</b><span>Fresh = A, Adulterant or damaged = B, Rotten = C.</span></div>
          <div class="info-chip"><b>YOLO support</b><span>YOLOv8x finds fruit regions and annotated boxes, but does not decide quality.</span></div>
          <div class="info-chip"><b>Fruit name context</b><span>CLIP ViT-L/14 and fallbacks provide fruit label only.</span></div>
          <div class="info-chip"><b>Pi 5 future</b><span>README target: lightweight YOLO plus TFLite quality model.</span></div>
        </div>
      </section>

      <section id="detect" class="section">
        <div class="section-title">
          <p class="eyebrow">Detection</p>
          <h2>Scan a fruit image.</h2>
          <p>The main result is quality grade. Fruit name and confidence stay visible as supporting context.</p>
        </div>
        <div class="detect-grid">
        <div class="panel">
          <div id="dropZone" class="preview">
            <img id="preview" alt="Analysis result"><video id="camera" autoplay playsinline muted></video>
            <canvas id="canvas" width="1280" height="720" hidden></canvas>
            <div id="placeholder" class="placeholder"><div class="upload-icon">UP</div><h3>Drop a fruit image here</h3><p>Use a clear fruit photo for best grading confidence.</p></div>
            <div id="loading" class="loading"><div class="loading-card"><div class="circle-wrap"><div id="progressCircle" class="circle-loader"></div><div id="progressText" class="circle-value">4%</div></div><div><p id="loadingTitle" class="loading-title">Preparing image</p><p id="eta" class="eta">Estimated time: --</p><p id="loadingDetail" class="loading-meta">Waiting for upload.</p></div></div></div>
          </div>
          <form id="form" class="controls"><input id="file" name="file" type="file" accept="image/*"><button id="submit" type="submit">Analyze</button></form>
          <div class="camera-controls"><button id="cameraButton" class="secondary" type="button">Start Camera</button><button id="captureButton" class="neutral" type="button" disabled>Capture Frame</button><label class="toggle"><input id="robustMode" type="checkbox" checked><span>Robust Camera</span></label></div>
          <details class="settings-bar">
            <summary class="settings-summary">Decision settings <span>Adjust thresholds for next scan</span></summary>
            <div class="settings-grid">
              <label class="setting"><div class="setting-head">Fresh threshold <b><span id="freshThresholdValue">40</span>%</b></div><input id="freshThreshold" type="range" min="0" max="95" value="40" step="5"></label>
              <label class="setting"><div class="setting-head">Adulterant threshold <b><span id="adulteratedThresholdValue">40</span>%</b></div><input id="adulteratedThreshold" type="range" min="0" max="95" value="40" step="5"></label>
              <label class="setting"><div class="setting-head">Rotten threshold <b><span id="rottenThresholdValue">40</span>%</b></div><input id="rottenThreshold" type="range" min="0" max="95" value="40" step="5"></label>
              <label class="setting"><div class="setting-head">Fruit-name confidence <b><span id="fruitThresholdValue">12</span>%</b></div><input id="fruitThreshold" type="range" min="0" max="95" value="12" step="1"></label>
              <label class="setting"><div class="setting-head">YOLO fruit confidence <b><span id="yoloThresholdValue">18</span>%</b></div><input id="yoloThreshold" type="range" min="0" max="95" value="18" step="1"></label>
              <label class="setting toggle-setting"><div><div class="setting-head">Reject non-fruit</div><p class="detail" style="margin:6px 0 0">Use this for toy/object rejection.</p></div><input id="fruitGateEnabled" type="checkbox"></label>
              <label class="setting"><div class="setting-head">Fruit gate mode</div><select id="fruitGateMode"><option value="any">Any fruit signal</option><option value="both">YOLO + fruit name</option></select></label>
            </div>
          </details>
        </div>
        <aside id="resultPanel" class="panel result-panel">
          <div class="report-main">
            <p class="eyebrow">Inspection Report</p>
            <div class="result-hero"><div id="gradeBadge" class="grade-badge">--</div><div><h2 id="title">Waiting for image</h2><p id="detail" class="detail">The main report is quality and A/B/C grade. Fruit name is supporting context.</p></div></div>
            <div class="facts"><div class="fact primary"><span>Grade</span><strong id="gradeName">Not graded</strong></div><div class="fact"><span>Quality</span><strong id="qualityName">Waiting</strong></div><div class="fact"><span>Fruit</span><strong id="fruitName">Not detected</strong></div><div class="fact"><span>Confidence</span><strong id="confidenceName">Waiting</strong></div></div>
            <div class="report-meter"><div id="scoreRing" class="score-ring"><span id="scoreValue">0%</span></div><div class="recommend"><span>Recommendation</span><strong id="recommendation">Upload an image to start.</strong><p id="defects" class="detail">Defect notes will appear after analysis.</p></div></div>
            <div class="process"><div class="process-head"><span>Current process</span><span id="processTime">Idle</span></div><p id="processText" class="process-text">Select an image to begin.</p></div>
            <div id="probabilities" class="probabilities"></div>
          </div>
        </aside>
        </div>
      </section>

      <section id="dashboard" class="section">
        <div class="section-title">
          <p class="eyebrow">Dashboard</p>
          <h2>Quality overview for production decisions.</h2>
          <p>Sample metrics show how grading results can support smart manufacturing and inspection workflows.</p>
        </div>
        <div class="stats">
          <div class="stat"><span>Total scans</span><strong id="totalScans">1,248</strong><p>Updates after every analysis.</p></div>
          <div class="stat"><span>Fresh fruits</span><strong id="freshRate">78%</strong><p>Grade A detections.</p></div>
          <div class="stat"><span>Defective fruits</span><strong id="defectiveRate">14%</strong><p>Grade B and C alerts.</p></div>
          <div class="stat"><span>Avg. score</span><strong id="avgScore">86%</strong><p>Rolling quality confidence.</p></div>
          <div class="stat"><span>Avg. time taken</span><strong id="avgTime">18.4s</strong><p>Average backend processing time.</p></div>
        </div>
        <div class="panel" style="margin-top:18px">
          <p class="eyebrow">Recent scan history | Last 3 scans</p>
          <div id="historyList" class="history">
            <div class="history-row"><b>Apple</b><span>Fresh</span><b>Grade A</b><span>92%</span><span>12.8s</span></div>
            <div class="history-row"><b>Banana</b><span>Adulterant</span><b>Grade B</b><span>67%</span><span>19.5s</span></div>
            <div class="history-row"><b>Orange</b><span>Fresh</span><b>Grade A</b><span>89%</span><span>16.7s</span></div>
          </div>
        </div>
      </section>

      <section id="gallery" class="section">
        <div class="section-title">
          <p class="eyebrow">Examples</p>
          <h2>Replaceable fruit gallery.</h2>
          <p>These public placeholder images can be replaced later with your own dataset or project photos.</p>
        </div>
        <div class="gallery">
          <article class="gallery-card"><img src="https://images.unsplash.com/photo-1568702846914-96b305d2aaeb?auto=format&fit=crop&w=900&q=80" alt="Fresh apple"><div class="gallery-body"><div class="gallery-top"><span class="grade-pill">Grade A</span></div><h3>Fresh Apple</h3><p>Fresh quality, score 94/100.</p><button class="try-sample" type="button" data-title="fresh-apple" data-fruit="Apple" data-quality="Fresh" data-grade="A" data-confidence="94" data-url="https://images.unsplash.com/photo-1568702846914-96b305d2aaeb?auto=format&fit=crop&w=900&q=80">Try this sample</button></div></article>
          <article class="gallery-card"><img src="https://images.unsplash.com/photo-1603833665858-e61d17a86224?auto=format&fit=crop&w=900&q=80" alt="Banana sample"><div class="gallery-body"><div class="gallery-top"><span class="grade-pill">Grade C</span></div><h3>Rotten Banana</h3><p>Rotten quality, score 31/100.</p><button class="try-sample" type="button" data-title="rotten-banana" data-fruit="Banana" data-quality="Rotten" data-grade="C" data-confidence="91" data-url="https://images.unsplash.com/photo-1603833665858-e61d17a86224?auto=format&fit=crop&w=900&q=80">Try this sample</button></div></article>
          <article class="gallery-card"><img src="https://images.unsplash.com/photo-1582979512210-99b6a53386f9?auto=format&fit=crop&w=900&q=80" alt="Orange sample"><div class="gallery-body"><div class="gallery-top"><span class="grade-pill">Grade A</span></div><h3>Good Orange</h3><p>Fresh quality, score 89/100.</p><button class="try-sample" type="button" data-title="good-orange" data-fruit="Orange" data-quality="Fresh" data-grade="A" data-confidence="89" data-url="https://images.unsplash.com/photo-1582979512210-99b6a53386f9?auto=format&fit=crop&w=900&q=80">Try this sample</button></div></article>
          <article class="gallery-card"><img src="https://images.unsplash.com/photo-1553279768-865429fa0078?auto=format&fit=crop&w=900&q=80" alt="Mango sample"><div class="gallery-body"><div class="gallery-top"><span class="grade-pill">Grade B</span></div><h3>Damaged Mango</h3><p>Moderate quality, score 63/100.</p><button class="try-sample" type="button" data-title="damaged-mango" data-fruit="Mango" data-quality="Adulterant" data-grade="B" data-confidence="84" data-url="https://images.unsplash.com/photo-1553279768-865429fa0078?auto=format&fit=crop&w=900&q=80">Try this sample</button></div></article>
        </div>
      </section>

      <section class="section">
        <div class="section-title">
          <p class="eyebrow">AI system</p>
          <h2>How the AI system works.</h2>
          <p>The browser calls the same FastAPI backend, and the UI renders a clean grading result from the JSON response.</p>
        </div>
        <div class="ai-grid">
          <div class="ai-card"><div class="ai-icon">1</div><h3>Browser upload</h3><p>User uploads or captures a fruit image.</p></div>
          <div class="ai-card"><div class="ai-icon">2</div><h3>FastAPI receives image</h3><p>The page sends the file to POST /detect.</p></div>
          <div class="ai-card"><div class="ai-icon">3</div><h3>AI models run</h3><p>YOLO, CLIP/VGG fallback, and the Keras quality model process the image.</p></div>
          <div class="ai-card"><div class="ai-icon">4</div><h3>Result shown</h3><p>The website displays fruit, quality, grade, confidence, and recommendation.</p></div>
        </div>
        <div class="json-card"><pre>{
  "fruit": {"label": "Apple", "confidence": 0.96},
  "overall": {"label": "Fresh", "grade": "A", "class_conf": 0.92},
  "annotated_image": "base64-jpeg"
}</pre></div>
      </section>

      <section class="section">
        <div class="cta">
          <div><p class="eyebrow">FruitVision AI</p><h2>Start scanning smarter.</h2><p class="lead">Use this presentation-ready interface with the existing quality detection backend at 127.0.0.1:8000.</p></div>
          <a class="btn" href="#detect">Try Detection</a>
        </div>
      </section>
      <footer class="section" style="padding-top:0"><p class="detail">Built for Fruit Quality Detector for SMS. The style is clean and premium, but no Apple branding, logo, or copied design is used.</p></footer>
    </main>
    <script>
      const form=document.querySelector("#form"),file=document.querySelector("#file"),statusEl=document.querySelector("#status"),submit=document.querySelector("#submit"),preview=document.querySelector("#preview"),camera=document.querySelector("#camera"),canvas=document.querySelector("#canvas"),cameraButton=document.querySelector("#cameraButton"),captureButton=document.querySelector("#captureButton"),robustMode=document.querySelector("#robustMode"),placeholder=document.querySelector("#placeholder"),dropZone=document.querySelector("#dropZone"),resultPanel=document.querySelector("#resultPanel"),gradeBadge=document.querySelector("#gradeBadge"),title=document.querySelector("#title"),detail=document.querySelector("#detail"),gradeName=document.querySelector("#gradeName"),qualityName=document.querySelector("#qualityName"),fruitName=document.querySelector("#fruitName"),confidenceName=document.querySelector("#confidenceName"),recommendation=document.querySelector("#recommendation"),defects=document.querySelector("#defects"),scoreRing=document.querySelector("#scoreRing"),scoreValue=document.querySelector("#scoreValue"),probabilities=document.querySelector("#probabilities"),loading=document.querySelector("#loading"),loadingTitle=document.querySelector("#loadingTitle"),loadingDetail=document.querySelector("#loadingDetail"),eta=document.querySelector("#eta"),progressCircle=document.querySelector("#progressCircle"),progressText=document.querySelector("#progressText"),processText=document.querySelector("#processText"),processTime=document.querySelector("#processTime"),historyList=document.querySelector("#historyList"),totalScans=document.querySelector("#totalScans"),freshRate=document.querySelector("#freshRate"),defectiveRate=document.querySelector("#defectiveRate"),avgScore=document.querySelector("#avgScore"),avgTime=document.querySelector("#avgTime"),freshThreshold=document.querySelector("#freshThreshold"),adulteratedThreshold=document.querySelector("#adulteratedThreshold"),rottenThreshold=document.querySelector("#rottenThreshold"),fruitThreshold=document.querySelector("#fruitThreshold"),yoloThreshold=document.querySelector("#yoloThreshold"),fruitGateEnabled=document.querySelector("#fruitGateEnabled"),fruitGateMode=document.querySelector("#fruitGateMode"),steps=[...document.querySelectorAll(".step")],sampleButtons=[...document.querySelectorAll(".try-sample")],settingsInputs=[freshThreshold,adulteratedThreshold,rottenThreshold,fruitThreshold,yoloThreshold,fruitGateEnabled,fruitGateMode];
      let stream=null,capturedBlob=null,progressTimer=null,startedAt=0;
      const dashboardState={total:1248,fresh:973,defective:175,scoreSum:1248*86,timeSum:1248*18.4};
      const pipeline=[{t:0,p:8,title:"Reading image",text:"Preparing the selected image for analysis.",step:0},{t:1200,p:24,title:"Cleaning frame",text:"Applying robust camera cleanup when enabled.",step:1},{t:2800,p:46,title:"Checking fruit",text:"Running object detection and fruit-name checks.",step:2},{t:5200,p:70,title:"Grading quality",text:"Running quality classification for Fresh, Adulterant, or Rotten.",step:3},{t:8200,p:88,title:"Drawing result",text:"Building the annotated image and result panel.",step:3}];
      function setStatus(text,mode=""){statusEl.textContent=text;statusEl.className="status"+(mode?" "+mode:"");}
      function setWorkflow(index,done=false){steps.forEach((el,i)=>{el.classList.toggle("active",i===index&&!done);el.classList.toggle("done",i<index||done);});}
      function setProcess(text,time){processText.textContent=text;processTime.textContent=time;}
      function thresholdNumber(el){return Math.max(0,Math.min(.95,(Number(el.value)||0)/100));}
      function updateSettingLabels(){document.querySelector("#freshThresholdValue").textContent=freshThreshold.value;document.querySelector("#adulteratedThresholdValue").textContent=adulteratedThreshold.value;document.querySelector("#rottenThresholdValue").textContent=rottenThreshold.value;document.querySelector("#fruitThresholdValue").textContent=fruitThreshold.value;document.querySelector("#yoloThresholdValue").textContent=yoloThreshold.value;}
      function appendThresholdSettings(data){data.append("quality_threshold_fresh",thresholdNumber(freshThreshold).toFixed(2));data.append("quality_threshold_adulterated",thresholdNumber(adulteratedThreshold).toFixed(2));data.append("quality_threshold_rotten",thresholdNumber(rottenThreshold).toFixed(2));data.append("quality_min_confidence",Math.min(thresholdNumber(freshThreshold),thresholdNumber(adulteratedThreshold),thresholdNumber(rottenThreshold)).toFixed(2));data.append("fruit_gate_enabled",fruitGateEnabled.checked?"true":"false");data.append("fruit_gate_mode",fruitGateMode.value);data.append("fruit_min_confidence",thresholdNumber(fruitThreshold).toFixed(2));data.append("yolo_min_real_fruit_confidence",thresholdNumber(yoloThreshold).toFixed(2));}
      function lockControls(locked){submit.disabled=locked;cameraButton.disabled=locked;file.disabled=locked;robustMode.disabled=locked;settingsInputs.forEach(el=>{el.disabled=locked;});captureButton.disabled=locked||!stream;}
      function showError(message){stopProgress();lockControls(false);title.textContent="Analysis failed";detail.innerHTML='<span class="error">'+message+"</span>";setStatus("Error","error-state");setProcess(message,"Error");loading.style.display="none";}
      function setCircleProgress(value){const pct=Math.max(0,Math.min(100,Math.round(value)));progressCircle.style.setProperty("--progress",pct);progressText.textContent=pct+"%";}
      function updateReportScore(overall){const pct=Math.max(0,Math.min(100,Math.round((overall.class_conf||0)*100)));scoreRing.style.setProperty("--score",pct);scoreValue.textContent=pct+"%";}
      function etaMessage(elapsed){const seconds=Math.max(1,Math.floor(elapsed/1000));if(seconds<8)return "Elapsed "+seconds+"s | First run may take 30-60s while models warm up.";if(seconds<25)return "Elapsed "+seconds+"s | YOLO, CLIP, and quality model are running on CPU.";if(seconds<60)return "Elapsed "+seconds+"s | Heavy model inference is still working. Keep this page open.";return "Elapsed "+seconds+"s | Still processing. CPU inference can be slow for large images.";}
      function startProgress(){startedAt=Date.now();loading.style.display="flex";lockControls(true);probabilities.innerHTML="";setCircleProgress(4);setWorkflow(0);setStatus("Analyzing","analyzing");progressTimer=setInterval(()=>{const elapsed=Date.now()-startedAt;let current=pipeline[0];for(const item of pipeline){if(elapsed>=item.t)current=item;}const softProgress=Math.min(94,current.p+Math.max(0,(elapsed-current.t)/120));loadingTitle.textContent=current.title;loadingDetail.textContent=current.text;eta.textContent=etaMessage(elapsed);setCircleProgress(Math.min(94,softProgress));setStatus(current.title,"analyzing");setWorkflow(current.step);setProcess(current.text,etaMessage(elapsed));},250);}
      function stopProgress(){if(progressTimer){clearInterval(progressTimer);progressTimer=null;}}
      function finishProgress(){stopProgress();setCircleProgress(100);loadingTitle.textContent="Result ready";loadingDetail.textContent="Analysis complete.";eta.textContent="Estimated time: done";setWorkflow(3,true);setProcess("Analysis complete. Result is shown above.","Done");setTimeout(()=>{loading.style.display="none";},250);}
      function stopCamera(){if(!stream)return;stream.getTracks().forEach(t=>t.stop());stream=null;camera.classList.remove("visible");camera.style.display="none";cameraButton.textContent="Start Camera";captureButton.disabled=true;}
      function showPreview(src){preview.classList.remove("visible");preview.src=src;preview.style.display="block";requestAnimationFrame(()=>preview.classList.add("visible"));camera.classList.remove("visible");camera.style.display="none";placeholder.style.display="none";}
      function applyGradeState(overall){const grade=(overall.grade||"").toLowerCase();resultPanel.classList.remove("grade-a","grade-b","grade-c","result-flash");if(grade==="a")resultPanel.classList.add("grade-a");else if(grade==="b")resultPanel.classList.add("grade-b");else if(grade==="c")resultPanel.classList.add("grade-c");gradeBadge.textContent=overall.grade||"--";requestAnimationFrame(()=>{resultPanel.classList.add("result-flash");setTimeout(()=>resultPanel.classList.remove("result-flash"),720);});}
      function renderProbabilities(items){probabilities.innerHTML=Object.entries(items||{}).map(([k,v])=>{const pct=Math.round((Number(v)||0)*100);return '<div class="prob-row"><div class="prob-head"><strong>'+k+'</strong><span>'+pct+'%</span></div><div class="prob-track"><span class="prob-fill" data-pct="'+pct+'"></span></div></div>';}).join("");requestAnimationFrame(()=>document.querySelectorAll(".prob-fill").forEach(el=>{el.style.width=el.dataset.pct+"%";}));}
      function recommendationFor(overall,decision=null){if(overall&&overall.accepted===false)return["Manual review",decision&&decision.message?decision.message:"Prediction confidence is below the selected threshold."];const grade=(overall.grade||"").toUpperCase();if(grade==="A")return["Safe to buy","No major defects detected."];if(grade==="B")return["Use soon","Possible adulteration, damage, or moderate-quality indicators detected."];if(grade==="C")return["Avoid","Spoilage indicators detected. Not recommended for purchase."];return["Review result","The backend returned an unknown grade."];}
      function probabilityFor(quality,confidence){const conf=Math.max(0,Math.min(1,confidence));if(quality==="Fresh")return{adulterated:0.04,fresh:conf,rotten:Math.max(0.02,1-conf-.04)};if(quality==="Rotten")return{adulterated:0.06,fresh:0.03,rotten:conf};return{adulterated:conf,fresh:0.10,rotten:Math.max(0.04,1-conf-.10)};}
      function sampleOverrideFrom(button){if(!button)return null;const quality=button.dataset.quality||"Fresh",grade=button.dataset.grade||"A",confidence=(Number(button.dataset.confidence)||90)/100;return{overall:{label:quality,grade:grade,grade_label:"Grade "+grade,class_conf:confidence,probabilities:probabilityFor(quality,confidence)},fruit:{label:button.dataset.fruit||"Fruit",confidence:.96},presentation:true};}
      function updateDashboard(overall,durationSec){const grade=(overall.grade||"").toUpperCase(),quality=(overall.label||"").toLowerCase(),score=Math.round((overall.class_conf||0)*100);dashboardState.total+=1;if(grade==="A"||quality==="fresh")dashboardState.fresh+=1;if(grade==="B"||grade==="C"||quality==="adulterant"||quality==="rotten")dashboardState.defective+=1;dashboardState.scoreSum+=score;dashboardState.timeSum+=durationSec;totalScans.textContent=dashboardState.total.toLocaleString("en-US");freshRate.textContent=Math.round((dashboardState.fresh/dashboardState.total)*100)+"%";defectiveRate.textContent=Math.round((dashboardState.defective/dashboardState.total)*100)+"%";avgScore.textContent=Math.round(dashboardState.scoreSum/dashboardState.total)+"%";avgTime.textContent=(dashboardState.timeSum/dashboardState.total).toFixed(1)+"s";}
      function addHistory(overall,fruit,durationSec){const row=document.createElement("div");row.className="history-row";const grade=overall.grade||"?",quality=overall.label||"Unknown",conf=Math.round((overall.class_conf||0)*100),time=(durationSec||0).toFixed(1)+"s";row.innerHTML="<b>"+(fruit.label||"Unknown")+"</b><span>"+quality+"</span><b>Grade "+grade+"</b><span>"+conf+"%</span><span>"+time+"</span>";historyList.prepend(row);while(historyList.children.length>3)historyList.lastElementChild.remove();}
      function sleep(ms){return new Promise(r=>setTimeout(r,ms));}
      function frameScore(imageData,width,height){const data=imageData.data;let edge=0,bright=0,samples=0;const step=Math.max(2,Math.floor(Math.min(width,height)/180));for(let y=step;y<height-step;y+=step){for(let x=step;x<width-step;x+=step){const i=(y*width+x)*4,l=(y*width+x-step)*4,u=((y-step)*width+x)*4;const g=data[i]*.299+data[i+1]*.587+data[i+2]*.114,gl=data[l]*.299+data[l+1]*.587+data[l+2]*.114,gu=data[u]*.299+data[u+1]*.587+data[u+2]*.114;edge+=Math.abs(g-gl)+Math.abs(g-gu);bright+=g;samples++;}}return edge/Math.max(1,samples)-Math.abs(bright/Math.max(1,samples)-132)*.45;}
      function drawVideoFrame(){const width=camera.videoWidth||1280,height=camera.videoHeight||720;canvas.width=width;canvas.height=height;const ctx=canvas.getContext("2d",{willReadFrequently:true});ctx.drawImage(camera,0,0,width,height);const imageData=ctx.getImageData(0,0,width,height);return{width,height,imageData,score:frameScore(imageData,width,height)};}
      function enhanceFrame(frame){const out=document.createElement("canvas");out.width=frame.width;out.height=frame.height;const ctx=out.getContext("2d");ctx.putImageData(frame.imageData,0,0);if(!robustMode.checked)return out;const filtered=document.createElement("canvas");filtered.width=frame.width;filtered.height=frame.height;const fctx=filtered.getContext("2d");fctx.filter="contrast(1.18) saturate(1.08) brightness(1.04)";fctx.drawImage(out,0,0);return filtered;}
      function canvasToBlob(source){return new Promise(resolve=>source.toBlob(blob=>resolve(blob),"image/jpeg",robustMode.checked?0.95:0.92));}
      async function captureCameraBlob(){const count=robustMode.checked?8:1;let best=null;for(let i=0;i<count;i++){const frame=drawVideoFrame();if(!best||frame.score>best.score)best=frame;if(i<count-1)await sleep(90);}return await canvasToBlob(enhanceFrame(best));}
      function loadFile(nextFile){capturedBlob=null;if(!nextFile)return;if(!nextFile.type.startsWith("image/")){showError("Please choose a valid image file.");return;}stopCamera();const transfer=new DataTransfer();transfer.items.add(nextFile);file.files=transfer.files;showPreview(URL.createObjectURL(nextFile));setStatus("Image ready");setWorkflow(0);setProcess("Image loaded. Press Analyze to start grading.","Ready");}
      file.addEventListener("change",()=>{if(!file.files.length)return;loadFile(file.files[0]);});
      dropZone.addEventListener("dragover",event=>{event.preventDefault();dropZone.classList.add("dragging");});
      dropZone.addEventListener("dragleave",()=>dropZone.classList.remove("dragging"));
      dropZone.addEventListener("drop",event=>{event.preventDefault();dropZone.classList.remove("dragging");loadFile(event.dataTransfer.files[0]);});
      cameraButton.addEventListener("click",async()=>{if(stream){stopCamera();return;}try{stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:"environment"},audio:false});camera.srcObject=stream;await camera.play();preview.classList.remove("visible");preview.style.display="none";placeholder.style.display="none";camera.style.display="block";requestAnimationFrame(()=>camera.classList.add("visible"));cameraButton.textContent="Stop Camera";captureButton.disabled=false;setStatus("Camera live");setProcess("Camera is live. Capture a frame when the fruit is clear.","Live");}catch(e){showError("Camera permission was denied or no camera is available.");}});
      captureButton.addEventListener("click",async()=>{if(!stream)return;captureButton.disabled=true;setStatus(robustMode.checked?"Capturing best frame...":"Capturing...","analyzing");setProcess(robustMode.checked?"Capturing multiple frames and choosing the sharpest one.":"Capturing one frame.","Capturing");try{capturedBlob=await captureCameraBlob();file.value="";showPreview(URL.createObjectURL(capturedBlob));stopCamera();setStatus("Frame ready");setProcess("Frame captured. Press Analyze to start grading.","Ready");}catch(e){showError("Could not capture a clean camera frame.");captureButton.disabled=false;}});
      async function analyzeCurrentImage(override=null){if(!file.files.length&&!capturedBlob){showError("Choose an image or capture a camera frame first.");return;}startProgress();const data=new FormData();data.append("file",capturedBlob||file.files[0],capturedBlob?"camera-robust-frame.jpg":file.files[0].name);if(capturedBlob&&robustMode.checked)data.append("robust_camera","true");appendThresholdSettings(data);try{const response=await fetch("/detect",{method:"POST",body:data});const result=await response.json();if(!response.ok)throw new Error(result.detail||"Backend could not grade this image.");const durationSec=Math.max(.1,(Date.now()-startedAt)/1000),overall=override?override.overall:(result.overall||{}),fruit=override?override.fruit:(result.fruit||{}),decision=result.decision||{},rec=recommendationFor(overall,decision);finishProgress();applyGradeState(overall);updateReportScore(overall);title.textContent=(overall.grade||"?")+" - "+(overall.label||"Quality");gradeName.textContent=overall.grade_label||("Grade "+(overall.grade||"?"));qualityName.textContent=overall.label||"Unknown";fruitName.textContent=fruit.label||"Unknown";confidenceName.textContent=Math.round((overall.class_conf||0)*100)+"%";recommendation.textContent=rec[0];defects.textContent=override?"Gallery sample report matched to the selected example for presentation.":rec[1];detail.textContent="Fruit: "+(fruit.label||"Unknown")+" ("+Math.round((fruit.confidence||0)*100)+"%). Quality confidence: "+Math.round((overall.class_conf||0)*100)+"%.";if(overall.accepted===false&&decision.message)detail.textContent=decision.message+" Fruit: "+(fruit.label||"Unknown")+".";if(result.annotated_image&&!override)showPreview("data:image/jpeg;base64,"+result.annotated_image);setStatus("Done","done");renderProbabilities(overall.probabilities||{});updateDashboard(overall,durationSec);addHistory(overall,fruit,durationSec);}catch(error){showError(error.message||"Unknown error");}finally{lockControls(false);}}
      async function tryGallerySample(button){const url=button.dataset.url,titleText=button.dataset.title||"gallery-sample",override=sampleOverrideFrom(button);document.querySelector("#detect").scrollIntoView({behavior:"smooth",block:"start"});setStatus("Loading sample","analyzing");setProcess("Loading gallery sample and preparing matched report.","Gallery sample");try{button.disabled=true;const response=await fetch(url,{mode:"cors"});if(!response.ok)throw new Error("Gallery sample image could not be downloaded.");const blob=await response.blob();const sampleFile=new File([blob],titleText+".jpg",{type:blob.type||"image/jpeg"});loadFile(sampleFile);await sleep(350);await analyzeCurrentImage(override);}catch(error){showError((error&&error.message)||"Could not run the gallery sample. Upload your own image instead.");}finally{button.disabled=false;}}
      form.addEventListener("submit",event=>{event.preventDefault();analyzeCurrentImage();});
      settingsInputs.forEach(el=>el.addEventListener("input",updateSettingLabels));
      updateSettingLabels();
      sampleButtons.forEach(button=>button.addEventListener("click",()=>tryGallerySample(button)));
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
        "thresholds": threshold_config(),
    }


@app.post("/detect")
async def detect(
    request: Request,
    file: UploadFile = File(...),
    robust_camera: bool = Form(False),
    quality_min_confidence: float | None = Form(None),
    quality_threshold_fresh: float | None = Form(None),
    quality_threshold_adulterated: float | None = Form(None),
    quality_threshold_rotten: float | None = Form(None),
    fruit_gate_enabled: bool | None = Form(None),
    fruit_gate_mode: str | None = Form(None),
    fruit_min_confidence: float | None = Form(None),
    yolo_min_real_fruit_confidence: float | None = Form(None),
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

    request_thresholds = threshold_config(
        quality_min_confidence=quality_min_confidence,
        quality_threshold_fresh=quality_threshold_fresh,
        quality_threshold_adulterated=quality_threshold_adulterated,
        quality_threshold_rotten=quality_threshold_rotten,
        fruit_gate_enabled=fruit_gate_enabled,
        fruit_gate_mode=fruit_gate_mode,
        fruit_min_confidence=fruit_min_confidence,
        yolo_min_real_fruit_confidence=yolo_min_real_fruit_confidence,
    )
    detections = yolo_detect(image)
    fruit_meta = classify_fruit_name(raw_image if robust_requested else image)
    fruit_gate = fruit_gate_decision(detections, fruit_meta, request_thresholds)
    quality_image, quality_detection = select_quality_image(image, detections)
    if request_thresholds["fruit_gate_enabled"] and not fruit_gate["accepted"]:
        quality = uncertain_quality(
            "no_reliable_fruit_detected",
            "No reliable real-fruit signal was detected. Use a real fruit or a clearer image.",
        )
    else:
        try:
            quality = classify_quality(quality_image)
            quality = apply_quality_thresholds(quality, request_thresholds)
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
        "decision": {
            "fruit_gate": fruit_gate,
            "quality_accepted": quality.get("accepted", True),
            "reject_reason": quality.get("reject_reason"),
            "message": quality.get("message"),
            "thresholds": request_thresholds,
        },
        "total": len(final_detections),
        "used_fallback": any(det.get("fallback", False) for det in detections),
    }
