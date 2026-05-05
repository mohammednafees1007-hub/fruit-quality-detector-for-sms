from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "fruit_type_training" / "imagefolder"
DEFAULT_KERAS_OUTPUT = PROJECT_ROOT / "models" / "fruit_type_classifier.keras"
DEFAULT_TFLITE_OUTPUT = PROJECT_ROOT / "models" / "fruit_type_classifier.tflite"
DEFAULT_CLASSES_OUTPUT = PROJECT_ROOT / "models" / "fruit_type_classes.json"


def load_split(data_dir: Path, split: str, image_size: int, batch_size: int, shuffle: bool):
    split_dir = data_dir / split
    if not split_dir.exists():
        raise RuntimeError(f"Missing split folder: {split_dir}")
    return tf.keras.utils.image_dataset_from_directory(
        split_dir,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=shuffle,
        seed=1337,
    )


def build_model(image_size: int, class_count: int, dropout: float) -> tf.keras.Model:
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.18),
            tf.keras.layers.RandomTranslation(0.08, 0.08),
            tf.keras.layers.RandomContrast(0.28),
            tf.keras.layers.RandomBrightness(0.16),
        ],
        name="phone_camera_augmentation",
    )
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet",
        alpha=0.75,
    )
    backbone.trainable = False
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="fruit_image")
    x = augmentation(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenetv2_preprocess")(x)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="fruit_type_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def strip_augmentation(model: tf.keras.Model) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=model.input_shape[1:], name="fruit_image")
    x = inputs
    for layer in model.layers[1:]:
        if layer.name == "phone_camera_augmentation":
            continue
        try:
            x = layer(x, training=False)
        except TypeError:
            x = layer(x)
    return tf.keras.Model(inputs, x, name="fruit_type_classifier_inference")


def export_tflite(model: tf.keras.Model, output: Path) -> None:
    inference_model = strip_augmentation(model)
    converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(converter.convert())


def evaluate(model: tf.keras.Model, test_ds, class_names: list[str]) -> dict:
    y_true = []
    y_pred = []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
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
    parser = argparse.ArgumentParser(description="Train a broad fruit-name classifier from Hugging Face fruit datasets.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--keras-output", type=Path, default=DEFAULT_KERAS_OUTPUT)
    parser.add_argument("--tflite-output", type=Path, default=DEFAULT_TFLITE_OUTPUT)
    parser.add_argument("--classes-output", type=Path, default=DEFAULT_CLASSES_OUTPUT)
    parser.add_argument("--metrics-output", type=Path, default=PROJECT_ROOT / "models" / "fruit_type_classifier.metrics.json")
    args = parser.parse_args()

    train_ds = load_split(args.data_dir, "train", args.image_size, args.batch_size, shuffle=True)
    val_ds = load_split(args.data_dir, "val", args.image_size, args.batch_size, shuffle=False)
    test_ds = load_split(args.data_dir, "test", args.image_size, args.batch_size, shuffle=False)
    class_names = list(train_ds.class_names)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    model = build_model(args.image_size, len(class_names), args.dropout)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.4, patience=1, min_lr=1e-5),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    metrics = {
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "classes": class_names,
        "history": {key: [round(float(v), 6) for v in values] for key, values in history.history.items()},
        "test": evaluate(model, test_ds, class_names),
    }

    args.keras_output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.keras_output)
    export_tflite(model, args.tflite_output)
    args.classes_output.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    args.metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics["test"], indent=2))
    print(f"Saved Keras model: {args.keras_output.resolve()}")
    print(f"Saved TFLite model: {args.tflite_output.resolve()}")
    print(f"Saved classes: {args.classes_output.resolve()}")
    print(f"Saved metrics: {args.metrics_output.resolve()}")


if __name__ == "__main__":
    main()
