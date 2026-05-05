from __future__ import annotations

import argparse
import csv
import html
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

import main as app_core


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CLASSES = ["adulterated", "fresh", "rotten"]
GRADE_BY_CLASS = {
    "fresh": "A",
    "adulterated": "B",
    "rotten": "C",
}
DISPLAY_BY_CLASS = {
    "fresh": "Fresh",
    "adulterated": "Adulterant",
    "rotten": "Rotten",
}


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def normalize_label(value: str | None) -> str | None:
    if not value:
        return None
    text = value.lower().strip().replace("-", "_").replace(" ", "_")
    if "fresh" in text:
        return "fresh"
    if "rot" in text:
        return "rotten"
    if any(token in text for token in ("adulter", "formalin", "damage", "damaged")):
        return "adulterated"
    if text in DEFAULT_CLASSES:
        return text
    return None


def display_label(label: str | None) -> str:
    if not label:
        return "Unknown"
    return DISPLAY_BY_CLASS.get(label, label.replace("_", " ").title())


def grade_for_label(label: str | None) -> str:
    return GRADE_BY_CLASS.get(label or "", "?")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def ms(value: float) -> str:
    return f"{value:.1f} ms"


def collect_images(input_path: Path) -> list[dict[str, Any]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    records: list[dict[str, Any]] = []
    for path in sorted(input_path.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        true_label = normalize_label(path.parent.name)
        fruit_hint = path.stem.lower().split("_")[0].split("-")[0]
        records.append({
            "path": path,
            "true_label": true_label,
            "fruit_hint": fruit_hint,
        })

    if not records:
        raise RuntimeError(f"No image files found under: {input_path}")
    return records


def balanced_sample(records: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or limit >= len(records):
        chosen = list(records)
        random.Random(seed).shuffle(chosen)
        return chosen

    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unlabeled: list[dict[str, Any]] = []
    for record in records:
        label = record.get("true_label")
        if label:
            by_class[str(label)].append(record)
        else:
            unlabeled.append(record)

    rng = random.Random(seed)
    for group in by_class.values():
        rng.shuffle(group)
    rng.shuffle(unlabeled)

    if not by_class:
        chosen = list(records)
        rng.shuffle(chosen)
        return chosen[:limit]

    labels = sorted(by_class)
    base = limit // len(labels)
    remainder = limit % len(labels)
    chosen: list[dict[str, Any]] = []
    leftovers: list[dict[str, Any]] = []

    for index, label in enumerate(labels):
        quota = base + (1 if index < remainder else 0)
        group = by_class[label]
        chosen.extend(group[:quota])
        leftovers.extend(group[quota:])

    if len(chosen) < limit:
        leftovers.extend(unlabeled)
        rng.shuffle(leftovers)
        chosen.extend(leftovers[: limit - len(chosen)])

    rng.shuffle(chosen)
    return chosen[:limit]


def image_info(path: Path, image: np.ndarray | None) -> dict[str, Any]:
    size_kb = path.stat().st_size / 1024.0 if path.exists() else 0.0
    if image is None:
        return {"width": 0, "height": 0, "file_size_kb": round(size_kb, 2)}
    height, width = image.shape[:2]
    return {"width": int(width), "height": int(height), "file_size_kb": round(size_kb, 2)}


def evaluate_image(
    record: dict[str, Any],
    mode: str,
    robust: bool,
    skip_fruit_name: bool,
) -> dict[str, Any]:
    path = Path(record["path"])
    row: dict[str, Any] = {
        "file": str(path),
        "file_name": path.name,
        "true_label": record.get("true_label") or "",
        "true_grade": grade_for_label(record.get("true_label")),
        "pred_label": "",
        "pred_display": "",
        "pred_grade": "",
        "correct": "",
        "confidence": 0.0,
        "fruit_label": "",
        "fruit_confidence": "",
        "fruit_source": "",
        "yolo_box_found": "",
        "yolo_class": "",
        "yolo_confidence": "",
        "error": "",
        "read_ms": 0.0,
        "preprocess_ms": 0.0,
        "yolo_ms": 0.0,
        "crop_ms": 0.0,
        "quality_ms": 0.0,
        "fruit_name_ms": 0.0,
        "total_ms": 0.0,
    }

    total_start = now_ms()
    try:
        read_start = now_ms()
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        row["read_ms"] = now_ms() - read_start
        row.update(image_info(path, image))
        if image is None:
            raise RuntimeError("OpenCV could not decode this image.")

        work_image = image
        if robust:
            preprocess_start = now_ms()
            work_image = app_core.robust_camera_preprocess(work_image)
            row["preprocess_ms"] = now_ms() - preprocess_start

        quality_image = work_image
        detections: list[dict[str, Any]] = []
        quality_detection: dict[str, Any] | None = None

        if mode == "full-pipeline":
            yolo_start = now_ms()
            detections = app_core.yolo_detect(work_image)
            row["yolo_ms"] = now_ms() - yolo_start

            crop_start = now_ms()
            quality_image, quality_detection = app_core.select_quality_image(work_image, detections)
            row["crop_ms"] = now_ms() - crop_start

            best_detection = quality_detection
            if best_detection is None and detections:
                best_detection = detections[0]
            if best_detection:
                row["yolo_box_found"] = not bool(best_detection.get("fallback", False))
                row["yolo_class"] = best_detection.get("yolo_class", "")
                row["yolo_confidence"] = best_detection.get("yolo_conf", "")

            if not skip_fruit_name:
                fruit_start = now_ms()
                fruit = app_core.classify_fruit_name(work_image)
                row["fruit_name_ms"] = now_ms() - fruit_start
                row["fruit_label"] = app_core.display_name(fruit.get("fruit_name"))
                row["fruit_confidence"] = fruit.get("fruit_conf", "")
                row["fruit_source"] = fruit.get("fruit_source", "")

        quality_start = now_ms()
        quality = app_core.classify_quality(quality_image)
        row["quality_ms"] = now_ms() - quality_start

        pred_label = normalize_label(quality.get("class")) or str(quality.get("class") or "")
        row["pred_label"] = pred_label
        row["pred_display"] = quality.get("label") or display_label(pred_label)
        row["pred_grade"] = quality.get("grade") or grade_for_label(pred_label)
        row["confidence"] = safe_float(quality.get("class_conf"))

        probabilities = quality.get("probabilities", {}) or {}
        for label in DEFAULT_CLASSES:
            row[f"prob_{label}"] = safe_float(probabilities.get(label), 0.0)

        true_label = record.get("true_label")
        row["correct"] = bool(true_label and pred_label == true_label)
    except Exception as exc:  # Keep a failed image in the report instead of stopping the batch.
        row["error"] = str(exc)
    finally:
        row["total_ms"] = now_ms() - total_start

    return row


def summarize(rows: list[dict[str, Any]], classes: list[str], mode: str, limit: int) -> dict[str, Any]:
    valid_rows = [row for row in rows if row.get("pred_label") and not row.get("error")]
    labeled_rows = [row for row in valid_rows if row.get("true_label")]
    y_true = [str(row["true_label"]) for row in labeled_rows]
    y_pred = [str(row["pred_label"]) for row in labeled_rows]

    summary: dict[str, Any] = {
        "mode": mode,
        "requested_limit": limit,
        "images_tested": len(rows),
        "valid_predictions": len(valid_rows),
        "failed_images": len(rows) - len(valid_rows),
        "class_order": classes,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if labeled_rows:
        correct = sum(1 for row in labeled_rows if row.get("correct") is True)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=classes,
            zero_division=0,
        )
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=classes,
            average="macro",
            zero_division=0,
        )
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=classes,
            average="weighted",
            zero_division=0,
        )
        matrix = confusion_matrix(y_true, y_pred, labels=classes)

        summary.update({
            "accuracy": correct / len(labeled_rows),
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
            "weighted_precision": float(weighted_precision),
            "weighted_recall": float(weighted_recall),
            "weighted_f1": float(weighted_f1),
            "per_class": {
                label: {
                    "precision": float(precision[index]),
                    "recall": float(recall[index]),
                    "f1": float(f1[index]),
                    "support": int(support[index]),
                }
                for index, label in enumerate(classes)
            },
            "confusion_matrix": matrix.astype(int).tolist(),
        })
    else:
        summary.update({
            "accuracy": None,
            "macro_precision": None,
            "macro_recall": None,
            "macro_f1": None,
            "weighted_precision": None,
            "weighted_recall": None,
            "weighted_f1": None,
            "per_class": {},
            "confusion_matrix": [],
        })

    timings = ["read_ms", "preprocess_ms", "yolo_ms", "crop_ms", "quality_ms", "fruit_name_ms", "total_ms"]
    timing_summary: dict[str, dict[str, float]] = {}
    for key in timings:
        values = [safe_float(row.get(key)) for row in valid_rows if safe_float(row.get(key)) >= 0]
        if values:
            timing_summary[key] = {
                "avg": float(statistics.mean(values)),
                "median": float(statistics.median(values)),
                "min": float(min(values)),
                "max": float(max(values)),
                "p95": float(np.percentile(values, 95)),
            }
    summary["timings_ms"] = timing_summary

    confidences = [safe_float(row.get("confidence")) for row in valid_rows]
    summary["confidence"] = {
        "avg": float(statistics.mean(confidences)) if confidences else 0.0,
        "median": float(statistics.median(confidences)) if confidences else 0.0,
        "min": float(min(confidences)) if confidences else 0.0,
        "max": float(max(confidences)) if confidences else 0.0,
    }
    summary["grade_distribution"] = dict(Counter(str(row.get("pred_grade") or "?") for row in valid_rows))
    summary["prediction_distribution"] = dict(Counter(str(row.get("pred_label") or "?") for row in valid_rows))
    summary["throughput_images_per_second"] = (
        1000.0 / timing_summary["total_ms"]["avg"]
        if timing_summary.get("total_ms", {}).get("avg")
        else 0.0
    )
    return summary


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "file_name",
        "file",
        "true_label",
        "true_grade",
        "pred_label",
        "pred_display",
        "pred_grade",
        "correct",
        "confidence",
        "prob_adulterated",
        "prob_fresh",
        "prob_rotten",
        "fruit_label",
        "fruit_confidence",
        "fruit_source",
        "yolo_box_found",
        "yolo_class",
        "yolo_confidence",
        "width",
        "height",
        "file_size_kb",
        "read_ms",
        "preprocess_ms",
        "yolo_ms",
        "crop_ms",
        "quality_ms",
        "fruit_name_ms",
        "total_ms",
        "error",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def svg_text(x: float, y: float, text: str, size: int = 14, fill: str = "#17231b", anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
        f'font-family="Segoe UI, Arial, sans-serif" text-anchor="{anchor}">{html.escape(text)}</text>'
    )


def svg_wrap(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" rx="22" fill="#fbfffc"/>'
        f"{body}</svg>"
    )


def write_svg(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def bar_chart(
    path: Path,
    title: str,
    labels: list[str],
    values: list[float],
    value_suffix: str = "",
    color: str = "#12824a",
    width: int = 920,
    height: int = 460,
) -> None:
    top = 78
    left = 84
    right = 36
    bottom = 92
    chart_w = width - left - right
    chart_h = height - top - bottom
    max_value = max(values) if values else 1.0
    max_value = max(max_value, 1.0)
    gap = 18
    bar_w = max(20, (chart_w - gap * max(0, len(values) - 1)) / max(1, len(values)))

    body = [
        svg_text(36, 42, title, 24, "#092015"),
        f'<line x1="{left}" y1="{top + chart_h}" x2="{width - right}" y2="{top + chart_h}" stroke="#d8e8dd" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#d8e8dd" stroke-width="2"/>',
    ]
    for tick in range(5):
        value = max_value * tick / 4
        y = top + chart_h - (value / max_value) * chart_h
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#eef5f0" stroke-width="1"/>')
        body.append(svg_text(left - 12, y + 5, f"{value:.0f}{value_suffix}", 12, "#617066", "end"))

    for index, (label, value) in enumerate(zip(labels, values)):
        x = left + index * (bar_w + gap)
        h = (value / max_value) * chart_h if max_value else 0
        y = top + chart_h - h
        body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="10" fill="{color}"/>')
        body.append(svg_text(x + bar_w / 2, y - 10, f"{value:.1f}{value_suffix}", 13, "#17331f", "middle"))
        body.append(svg_text(x + bar_w / 2, top + chart_h + 28, label.title(), 13, "#314239", "middle"))

    write_svg(path, svg_wrap(width, height, "".join(body)))


def grouped_bar_chart(
    path: Path,
    title: str,
    labels: list[str],
    series: dict[str, list[float]],
    width: int = 920,
    height: int = 500,
) -> None:
    colors = ["#12824a", "#d49b20", "#cf3d55", "#256f94"]
    top = 86
    left = 82
    right = 36
    bottom = 104
    chart_w = width - left - right
    chart_h = height - top - bottom
    names = list(series)
    max_value = max([1.0] + [value for values in series.values() for value in values])
    group_w = chart_w / max(1, len(labels))
    bar_w = max(14, (group_w - 24) / max(1, len(names)))

    body = [
        svg_text(36, 42, title, 24, "#092015"),
        f'<line x1="{left}" y1="{top + chart_h}" x2="{width - right}" y2="{top + chart_h}" stroke="#d8e8dd" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#d8e8dd" stroke-width="2"/>',
    ]
    for tick in range(6):
        value = max_value * tick / 5
        y = top + chart_h - (value / max_value) * chart_h
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#eef5f0" stroke-width="1"/>')
        body.append(svg_text(left - 12, y + 5, f"{value * 100:.0f}%", 12, "#617066", "end"))

    for group_index, label in enumerate(labels):
        group_x = left + group_index * group_w + 12
        for series_index, name in enumerate(names):
            value = series[name][group_index]
            x = group_x + series_index * bar_w
            h = (value / max_value) * chart_h
            y = top + chart_h - h
            body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 4:.1f}" height="{h:.1f}" rx="8" fill="{colors[series_index % len(colors)]}"/>')
        body.append(svg_text(group_x + (bar_w * len(names)) / 2, top + chart_h + 30, label.title(), 13, "#314239", "middle"))

    legend_x = left
    legend_y = height - 38
    for index, name in enumerate(names):
        x = legend_x + index * 145
        body.append(f'<rect x="{x}" y="{legend_y - 14}" width="16" height="16" rx="4" fill="{colors[index % len(colors)]}"/>')
        body.append(svg_text(x + 24, legend_y, name.title(), 13, "#314239"))

    write_svg(path, svg_wrap(width, height, "".join(body)))


def confusion_matrix_chart(
    path: Path,
    title: str,
    labels: list[str],
    matrix: list[list[int]],
    width: int = 760,
    height: int = 680,
) -> None:
    top = 108
    left = 170
    cell = 118
    max_value = max([1] + [int(value) for row in matrix for value in row])
    body = [
        svg_text(36, 42, title, 24, "#092015"),
        svg_text(left + cell * len(labels) / 2, 82, "Predicted class", 15, "#4e6256", "middle"),
        svg_text(42, top + cell * len(labels) / 2, "Actual class", 15, "#4e6256", "middle"),
    ]

    for index, label in enumerate(labels):
        body.append(svg_text(left + index * cell + cell / 2, top - 18, label.title(), 13, "#314239", "middle"))
        body.append(svg_text(left - 18, top + index * cell + cell / 2 + 5, label.title(), 13, "#314239", "end"))

    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            intensity = value / max_value
            green = int(245 - intensity * 145)
            red = int(232 - intensity * 112)
            blue = int(236 - intensity * 176)
            fill = f"rgb({red},{green},{blue})"
            x = left + col_index * cell
            y = top + row_index * cell
            body.append(f'<rect x="{x}" y="{y}" width="{cell - 4}" height="{cell - 4}" rx="18" fill="{fill}" stroke="#d9eadf"/>')
            body.append(svg_text(x + cell / 2, y + cell / 2 + 8, str(value), 28, "#092015", "middle"))

    write_svg(path, svg_wrap(width, height, "".join(body)))


def histogram_chart(path: Path, title: str, values: list[float]) -> None:
    buckets = [0] * 10
    for value in values:
        idx = min(9, max(0, int(value * 10)))
        buckets[idx] += 1
    labels = [f"{i * 10}-{(i + 1) * 10}" for i in range(10)]
    bar_chart(path, title, labels, [float(v) for v in buckets], "", "#1f7a62", width=980, height=460)


def latency_chart(path: Path, title: str, rows: list[dict[str, Any]], width: int = 980, height: int = 430) -> None:
    values = [safe_float(row.get("total_ms")) for row in rows if not row.get("error")]
    if not values:
        values = [0.0]
    top = 76
    left = 76
    right = 36
    bottom = 74
    chart_w = width - left - right
    chart_h = height - top - bottom
    max_value = max(values + [1.0])
    step = chart_w / max(1, len(values) - 1)

    points = []
    for index, value in enumerate(values):
        x = left + index * step
        y = top + chart_h - (value / max_value) * chart_h
        points.append(f"{x:.1f},{y:.1f}")

    body = [
        svg_text(36, 42, title, 24, "#092015"),
        f'<line x1="{left}" y1="{top + chart_h}" x2="{width - right}" y2="{top + chart_h}" stroke="#d8e8dd" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#d8e8dd" stroke-width="2"/>',
        f'<polyline points="{" ".join(points)}" fill="none" stroke="#12824a" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>',
    ]
    for tick in range(5):
        value = max_value * tick / 4
        y = top + chart_h - (value / max_value) * chart_h
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#eef5f0" stroke-width="1"/>')
        body.append(svg_text(left - 10, y + 5, f"{value:.0f} ms", 12, "#617066", "end"))
    body.append(svg_text(left, height - 28, "Image order", 13, "#617066"))
    body.append(svg_text(width - right, height - 28, f"{len(values)} images", 13, "#617066", "end"))
    write_svg(path, svg_wrap(width, height, "".join(body)))


def make_graphs(summary: dict[str, Any], rows: list[dict[str, Any]], output_dir: Path) -> dict[str, str]:
    graphs_dir = output_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    classes = list(summary["class_order"])
    graphs: dict[str, str] = {}

    matrix = summary.get("confusion_matrix") or [[0 for _ in classes] for _ in classes]
    confusion_path = graphs_dir / "confusion_matrix.svg"
    confusion_matrix_chart(confusion_path, "Confusion Matrix", classes, matrix)
    graphs["confusion_matrix"] = str(confusion_path.name if confusion_path.parent == output_dir else Path("graphs") / confusion_path.name)

    per_class = summary.get("per_class") or {}
    metrics_path = graphs_dir / "per_class_metrics.svg"
    grouped_bar_chart(
        metrics_path,
        "Per-Class Precision, Recall, and F1",
        classes,
        {
            "precision": [safe_float(per_class.get(label, {}).get("precision")) for label in classes],
            "recall": [safe_float(per_class.get(label, {}).get("recall")) for label in classes],
            "f1": [safe_float(per_class.get(label, {}).get("f1")) for label in classes],
        },
    )
    graphs["per_class_metrics"] = str(Path("graphs") / metrics_path.name)

    timing = summary.get("timings_ms", {})
    timing_labels = []
    timing_values = []
    for key in ["read_ms", "preprocess_ms", "yolo_ms", "crop_ms", "quality_ms", "fruit_name_ms"]:
        value = safe_float(timing.get(key, {}).get("avg"))
        if value > 0 or key in {"read_ms", "quality_ms"}:
            timing_labels.append(key.replace("_ms", "").replace("_", " "))
            timing_values.append(value)
    timing_path = graphs_dir / "timing_breakdown.svg"
    bar_chart(timing_path, "Average Timing Breakdown", timing_labels, timing_values, " ms", "#12824a")
    graphs["timing_breakdown"] = str(Path("graphs") / timing_path.name)

    confidence_path = graphs_dir / "confidence_histogram.svg"
    histogram_chart(confidence_path, "Confidence Distribution", [safe_float(row.get("confidence")) for row in rows if not row.get("error")])
    graphs["confidence_histogram"] = str(Path("graphs") / confidence_path.name)

    grade_distribution = summary.get("grade_distribution") or {}
    grade_labels = ["A", "B", "C", "?"]
    grade_values = [float(grade_distribution.get(label, 0)) for label in grade_labels]
    grade_path = graphs_dir / "grade_distribution.svg"
    bar_chart(grade_path, "Predicted Grade Distribution", grade_labels, grade_values, "", "#2c9160")
    graphs["grade_distribution"] = str(Path("graphs") / grade_path.name)

    latency_path = graphs_dir / "latency_per_image.svg"
    latency_chart(latency_path, "Latency Per Image", rows)
    graphs["latency_per_image"] = str(Path("graphs") / latency_path.name)

    return graphs


def metric_card(label: str, value: str) -> str:
    return (
        '<div class="card">'
        f'<div class="card-label">{html.escape(label)}</div>'
        f'<div class="card-value">{html.escape(value)}</div>'
        '</div>'
    )


def write_html_report(
    output_path: Path,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    graphs: dict[str, str],
) -> None:
    accuracy = summary.get("accuracy")
    macro_f1 = summary.get("macro_f1")
    avg_time = safe_float(summary.get("timings_ms", {}).get("total_ms", {}).get("avg"))
    p95_time = safe_float(summary.get("timings_ms", {}).get("total_ms", {}).get("p95"))
    avg_conf = safe_float(summary.get("confidence", {}).get("avg"))

    per_class_rows = []
    for label, metrics in (summary.get("per_class") or {}).items():
        per_class_rows.append(
            "<tr>"
            f"<td>{html.escape(label.title())}</td>"
            f"<td>{pct(safe_float(metrics.get('precision')))}</td>"
            f"<td>{pct(safe_float(metrics.get('recall')))}</td>"
            f"<td>{pct(safe_float(metrics.get('f1')))}</td>"
            f"<td>{int(metrics.get('support', 0))}</td>"
            "</tr>"
        )

    sample_rows = []
    for row in rows[:50]:
        status = "Pass" if row.get("correct") is True else "Fail"
        if row.get("error"):
            status = "Error"
        sample_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('file_name', '')))}</td>"
            f"<td>{html.escape(display_label(str(row.get('true_label') or '')))}</td>"
            f"<td>{html.escape(display_label(str(row.get('pred_label') or '')))}</td>"
            f"<td>{html.escape(str(row.get('pred_grade') or ''))}</td>"
            f"<td>{pct(safe_float(row.get('confidence')))}</td>"
            f"<td>{ms(safe_float(row.get('total_ms')))}</td>"
            f"<td><span class=\"pill {status.lower()}\">{status}</span></td>"
            "</tr>"
        )

    graph_imgs = "\n".join(
        f'<section class="panel"><img src="{html.escape(src)}" alt="{html.escape(name)}"></section>'
        for name, src in graphs.items()
    )
    model_profile = summary.get("model_profile", {}) or {}
    dataset_profile = summary.get("dataset", {}) or {}
    model_rows = "\n".join([
        f"<tr><td>Quality model</td><td>{html.escape(str(model_profile.get('quality_model_path', '')))}</td></tr>",
        f"<tr><td>Model parameters</td><td>{int(model_profile.get('quality_model_params', 0)):,}</td></tr>",
        f"<tr><td>Input shape</td><td>{html.escape(str(model_profile.get('quality_model_input_shape', '')))}</td></tr>",
        f"<tr><td>Output shape</td><td>{html.escape(str(model_profile.get('quality_model_output_shape', '')))}</td></tr>",
        f"<tr><td>Classes</td><td>{html.escape(', '.join(model_profile.get('classes', [])))}</td></tr>",
        f"<tr><td>Dataset input</td><td>{html.escape(str(dataset_profile.get('input_path', '')))}</td></tr>",
        f"<tr><td>Sample seed</td><td>{html.escape(str(dataset_profile.get('seed', '')))}</td></tr>",
    ])

    cards = "\n".join([
        metric_card("Images tested", str(summary.get("images_tested", 0))),
        metric_card("Accuracy", "N/A" if accuracy is None else pct(float(accuracy))),
        metric_card("Macro F1", "N/A" if macro_f1 is None else pct(float(macro_f1))),
        metric_card("Average confidence", pct(avg_conf)),
        metric_card("Average time", ms(avg_time)),
        metric_card("P95 time", ms(p95_time)),
    ])

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fruit Quality Model Master Test Report</title>
  <style>
    :root {{
      --ink: #09150f;
      --muted: #5e7065;
      --line: #dceade;
      --leaf: #12824a;
      --paper: #f4faf6;
      --card: rgba(255,255,255,.82);
      --shadow: 0 24px 80px rgba(6, 40, 20, .11);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 12% 0%, rgba(18,130,74,.16), transparent 28%),
        linear-gradient(135deg, #f6fbf7 0%, #ffffff 42%, #e8f6ee 100%);
    }}
    .wrap {{ width: min(1180px, calc(100% - 34px)); margin: 0 auto; padding: 42px 0 64px; }}
    .hero {{ padding: 42px; border: 1px solid rgba(255,255,255,.8); border-radius: 34px; background: var(--card); box-shadow: var(--shadow); backdrop-filter: blur(18px); }}
    .eyebrow {{ color: var(--leaf); font-weight: 900; text-transform: uppercase; letter-spacing: .08em; font-size: .8rem; }}
    h1 {{ margin: 12px 0 12px; font-size: clamp(2.2rem, 6vw, 4.8rem); line-height: .96; letter-spacing: 0; }}
    .lead {{ max-width: 760px; color: var(--muted); font-size: 1.1rem; line-height: 1.7; }}
    .cards {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 14px; margin: 22px 0; }}
    .card {{ grid-column: span 2; min-height: 114px; padding: 20px; border: 1px solid var(--line); border-radius: 22px; background: rgba(255,255,255,.86); box-shadow: 0 14px 34px rgba(6, 40, 20, .06); }}
    .card-label {{ color: var(--muted); font-weight: 800; text-transform: uppercase; font-size: .75rem; }}
    .card-value {{ margin-top: 10px; font-size: 1.8rem; font-weight: 950; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 22px; }}
    .panel {{ padding: 18px; border: 1px solid var(--line); border-radius: 26px; background: rgba(255,255,255,.8); box-shadow: 0 16px 40px rgba(6, 40, 20, .07); overflow: hidden; }}
    .panel img {{ width: 100%; display: block; border-radius: 18px; }}
    h2 {{ margin: 34px 0 14px; font-size: 1.45rem; }}
    table {{ width: 100%; border-collapse: collapse; overflow: hidden; border-radius: 18px; background: white; }}
    th, td {{ padding: 13px 14px; border-bottom: 1px solid #e8f0ea; text-align: left; font-size: .92rem; }}
    th {{ color: #415046; font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; }}
    .pill {{ display: inline-flex; align-items: center; padding: 6px 10px; border-radius: 999px; font-weight: 900; font-size: .78rem; }}
    .pill.pass {{ background: #e3f6e9; color: #0f6c3a; }}
    .pill.fail {{ background: #fff3dc; color: #9a6508; }}
    .pill.error {{ background: #ffe8ea; color: #b31d34; }}
    .links {{ display:flex; gap: 12px; flex-wrap: wrap; margin-top: 18px; }}
    .links a {{ color: white; background: var(--leaf); padding: 11px 14px; border-radius: 999px; text-decoration: none; font-weight: 850; }}
    @media (max-width: 860px) {{
      .cards, .grid {{ grid-template-columns: 1fr; }}
      .card {{ grid-column: span 1; }}
      .hero {{ padding: 26px; }}
    }}
  </style>
</head>
<body>
  <main class="wrap">
    <section class="hero">
      <div class="eyebrow">Master Model Test Report</div>
      <h1>Fruit quality grading performance.</h1>
      <p class="lead">
        This report tests the deployed fruit quality model on a controlled image sample.
        It measures grade accuracy, per-class precision, recall, F1 score, confidence,
        prediction distribution, and inference timing.
      </p>
      <div class="links">
        <a href="results.csv">Open CSV</a>
        <a href="summary.json">Open JSON Summary</a>
      </div>
    </section>

    <section class="cards">{cards}</section>

    <h2>Model and Test Setup</h2>
    <section class="panel">
      <table>
        <tbody>{model_rows}</tbody>
      </table>
    </section>

    <h2>Graphs</h2>
    <section class="grid">{graph_imgs}</section>

    <h2>Per-Class Metrics</h2>
    <section class="panel">
      <table>
        <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
        <tbody>{''.join(per_class_rows)}</tbody>
      </table>
    </section>

    <h2>Image Results</h2>
    <section class="panel">
      <table>
        <thead><tr><th>Image</th><th>Actual</th><th>Predicted</th><th>Grade</th><th>Confidence</th><th>Total Time</th><th>Status</th></tr></thead>
        <tbody>{''.join(sample_rows)}</tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""
    output_path.write_text(html_doc, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Master performance test for the Fruit Quality Detector model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data") / "quality_training" / "imagefolder" / "test",
        help="Image folder to evaluate. Parent folder names are used as true labels.",
    )
    parser.add_argument("--limit", type=int, default=50, help="Number of images to test.")
    parser.add_argument("--seed", type=int, default=1337, help="Deterministic sampling seed.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports") / "master_model_test",
        help="Output folder for CSV, JSON, HTML, and graphs.",
    )
    parser.add_argument(
        "--mode",
        choices=["quality-only", "full-pipeline"],
        default="quality-only",
        help="quality-only tests the grading model directly. full-pipeline also runs YOLO crop and optional fruit naming.",
    )
    parser.add_argument(
        "--robust",
        action="store_true",
        help="Apply the same robust camera preprocessing used by the website.",
    )
    parser.add_argument(
        "--skip-fruit-name",
        action="store_true",
        help="In full-pipeline mode, skip CLIP/VGG/MobileNet fruit naming to reduce test time.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = app_core.get_quality_model()
    if model is None:
        raise RuntimeError(
            f"Quality model is missing at {app_core.CLASSIFIER_PATH}. "
            "Train or restore the model before running the master test."
        )

    records = collect_images(args.input)
    sample = balanced_sample(records, args.limit, args.seed)
    classes = list(app_core.get_quality_classes() or DEFAULT_CLASSES)

    print(f"Testing {len(sample)} images from {args.input}")
    print(f"Mode: {args.mode}")
    print(f"Output: {output_dir}")

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(sample, start=1):
        row = evaluate_image(record, args.mode, args.robust, args.skip_fruit_name)
        rows.append(row)
        label = row.get("pred_label") or "error"
        confidence = safe_float(row.get("confidence"))
        print(f"[{index:02d}/{len(sample):02d}] {Path(record['path']).name} -> {label} ({confidence:.2%})")

    summary = summarize(rows, classes, args.mode, args.limit)
    summary["model_profile"] = {
        "quality_model_path": str(app_core.CLASSIFIER_PATH),
        "quality_model_input_shape": str(getattr(model, "input_shape", "")),
        "quality_model_output_shape": str(getattr(model, "output_shape", "")),
        "quality_model_params": int(model.count_params()),
        "quality_classes_path": str(app_core.QUALITY_CLASSES_PATH),
        "classes": classes,
        "yolo_model_path": str(app_core.YOLO_MODEL_PATH),
    }
    summary["dataset"] = {
        "input_path": str(args.input.resolve()),
        "available_images": len(records),
        "sampled_images": len(sample),
        "seed": args.seed,
    }

    csv_path = output_dir / "results.csv"
    json_path = output_dir / "summary.json"
    html_path = output_dir / "report.html"

    write_csv(rows, csv_path)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    graphs = make_graphs(summary, rows, output_dir)
    write_html_report(html_path, summary, rows, graphs)

    print("")
    print("Master test complete.")
    if summary.get("accuracy") is not None:
        print(f"Accuracy: {pct(float(summary['accuracy']))}")
        print(f"Macro F1: {pct(float(summary['macro_f1']))}")
    print(f"Average confidence: {pct(safe_float(summary.get('confidence', {}).get('avg')))}")
    print(f"Average total time: {ms(safe_float(summary.get('timings_ms', {}).get('total_ms', {}).get('avg')))}")
    print(f"HTML report: {html_path}")
    print(f"CSV results: {csv_path}")
    print(f"JSON summary: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
