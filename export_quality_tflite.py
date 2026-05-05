from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "models" / "fruit_quality_grader.keras"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "fruit_quality_grader.tflite"


def call_layer(layer, value):
    try:
        return layer(value, training=False)
    except TypeError:
        return layer(value)


def build_inference_model(model: tf.keras.Model, strip_augmentation: bool) -> tf.keras.Model:
    input_shape = model.input_shape[1:]
    inputs = tf.keras.Input(shape=input_shape, name="quality_image")
    x = inputs
    for layer in model.layers[1:]:
        if strip_augmentation and layer.name == "camera_robust_augmentation":
            continue
        x = call_layer(layer, x)
    return tf.keras.Model(inputs=inputs, outputs=x, name="fruit_quality_grader_tflite")


def export_tflite(input_model: Path, output_model: Path, optimize: bool, strip_augmentation: bool) -> None:
    if not input_model.exists():
        raise FileNotFoundError(f"Keras quality model not found: {input_model}")

    keras_model = tf.keras.models.load_model(input_model, compile=False)
    inference_model = build_inference_model(keras_model, strip_augmentation=strip_augmentation)
    converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
    if optimize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_bytes = converter.convert()

    output_model.parent.mkdir(parents=True, exist_ok=True)
    output_model.write_bytes(tflite_bytes)

    interpreter = tf.lite.Interpreter(model_path=str(output_model))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    print(f"Saved TFLite model: {output_model}")
    print(f"Input: shape={input_detail['shape'].tolist()} dtype={input_detail['dtype']}")
    print(f"Output: shape={output_detail['shape'].tolist()} dtype={output_detail['dtype']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the fruit quality Keras model to TensorFlow Lite.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-optimize", action="store_true", help="Disable TensorFlow Lite default optimization.")
    parser.add_argument("--keep-augmentation", action="store_true", help="Keep training augmentation layers in the exported graph.")
    args = parser.parse_args()

    export_tflite(
        input_model=args.input,
        output_model=args.output,
        optimize=not args.no_optimize,
        strip_augmentation=not args.keep_augmentation,
    )


if __name__ == "__main__":
    main()
