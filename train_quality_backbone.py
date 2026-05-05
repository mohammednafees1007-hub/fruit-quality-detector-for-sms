from __future__ import annotations

import argparse
import json
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score


PROJECT_ROOT = Path(__file__).resolve().parent


def make_backbone(name: str, image_size: int):
    input_shape = (image_size, image_size, 3)
    normalized = name.strip().lower()
    if normalized == "efficientnetv2b0":
        return keras.applications.EfficientNetV2B0(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            include_preprocessing=True,
        )
    if normalized == "efficientnetv2s":
        return keras.applications.EfficientNetV2S(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            include_preprocessing=True,
        )
    if normalized == "convnexttiny":
        return keras.applications.ConvNeXtTiny(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            include_preprocessing=True,
        )
    if normalized == "mobilenetv2":
        return keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
        )
    raise ValueError("Use efficientnetv2b0, efficientnetv2s, convnexttiny, or mobilenetv2.")


def build_model(backbone_name: str, image_size: int, class_count: int, dropout: float) -> keras.Model:
    backbone = make_backbone(backbone_name, image_size)
    backbone.trainable = False
    augmentation = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.05),
            keras.layers.RandomZoom(0.10),
            keras.layers.RandomContrast(0.20),
            keras.layers.RandomBrightness(0.12),
        ],
        name="camera_robust_augmentation",
    )
    inputs = keras.Input(shape=(image_size, image_size, 3))
    x = augmentation(inputs)
    if backbone_name.strip().lower() == "mobilenetv2":
        x = keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenetv2_preprocess")(x)
    x = backbone(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(class_count, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def unfreeze_top_layers(model: keras.Model, layer_count: int, learning_rate: float) -> None:
    backbones = [layer for layer in model.layers if isinstance(layer, keras.Model) and layer.name != "camera_robust_augmentation"]
    if not backbones:
        raise RuntimeError("Could not find backbone model to fine-tune.")
    backbone = backbones[0]
    backbone.trainable = True
    for layer in backbone.layers[:-layer_count]:
        layer.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def load_split(data_dir: Path, split: str, image_size: int, batch_size: int, shuffle: bool):
    split_dir = data_dir / split
    if not split_dir.exists():
        raise RuntimeError(f"Missing split folder: {split_dir}")
    return keras.utils.image_dataset_from_directory(
        split_dir,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=shuffle,
        seed=1337,
    )


def evaluate_model(model: keras.Model, test_ds, class_names: list[str]) -> dict:
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[list[float]] = []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        y_prob.extend(probs.tolist())
        y_true.extend(np.argmax(labels.numpy(), axis=1).tolist())
        y_pred.extend(np.argmax(probs, axis=1).tolist())

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "per_class": {
            name: {
                "precision": round(float(report[name]["precision"]), 4),
                "recall": round(float(report[name]["recall"]), 4),
                "f1": round(float(report[name]["f1-score"]), 4),
                "support": int(report[name]["support"]),
            }
            for name in class_names
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a quality-first fruit grading model.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "quality_training" / "imagefolder")
    parser.add_argument("--backbone", default="efficientnetv2b0")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--fine-tune-epochs", type=int, default=3)
    parser.add_argument("--fine-tune-layers", type=int, default=40)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "models" / "fruit_quality_grader.keras")
    parser.add_argument("--classes-output", type=Path, default=PROJECT_ROOT / "models" / "quality_classes.json")
    parser.add_argument("--metrics-output", type=Path, default=None)
    args = parser.parse_args()

    train_ds = load_split(args.data_dir, "train", args.image_size, args.batch_size, shuffle=True)
    val_ds = load_split(args.data_dir, "val", args.image_size, args.batch_size, shuffle=False)
    test_ds = load_split(args.data_dir, "test", args.image_size, args.batch_size, shuffle=False)
    class_names = list(train_ds.class_names)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.classes_output.parent.mkdir(parents=True, exist_ok=True)
    args.classes_output.write_text(json.dumps(class_names, indent=2), encoding="utf-8")

    model = build_model(args.backbone, args.image_size, len(class_names), args.dropout)
    callbacks = [
        keras.callbacks.ModelCheckpoint(args.output, monitor="val_accuracy", mode="max", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.4, patience=2, min_lr=1e-6),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    if args.fine_tune_epochs > 0:
        unfreeze_top_layers(model, args.fine_tune_layers, learning_rate=1e-5)
        fine_history = model.fit(train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs, callbacks=callbacks)
        for key, values in fine_history.history.items():
            history.history.setdefault(key, []).extend(values)

    best_model = keras.models.load_model(args.output, compile=False) if args.output.exists() else model
    metrics = {
        "backbone": args.backbone,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "classes": class_names,
        "history": {key: [round(float(item), 6) for item in values] for key, values in history.history.items()},
        "test": evaluate_model(best_model, test_ds, class_names),
    }
    metrics_output = args.metrics_output or args.output.with_suffix(".metrics.json")
    metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    best_model.save(args.output)
    print(json.dumps(metrics["test"], indent=2))
    print(f"Saved quality model: {args.output.resolve()}")
    print(f"Saved classes: {args.classes_output.resolve()}")
    print(f"Saved metrics: {metrics_output.resolve()}")


if __name__ == "__main__":
    main()
