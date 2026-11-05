"""Export the trained PyTorch model to a quantized TFLite model for fully
offline, on-device inference.

Pipeline: PyTorch (.pt) -> ONNX (.onnx) -> TensorFlow SavedModel -> TFLite (int8)

Run: python -m src.quantize
"""
import numpy as np
import onnx
import tensorflow as tf
import torch
from onnx_tf.backend import prepare

from src.config import (IMG_SIZE, ONNX_PATH, TFLITE_FP32_PATH, TFLITE_INT8_PATH)
from src.gradcam import load_trained_model


def export_onnx(model, onnx_path=ONNX_PATH):
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy_input, str(onnx_path),
        input_names=["input"], output_names=["output"],
        opset_version=13,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    print(f"Saved ONNX model to {onnx_path}")


def onnx_to_saved_model(onnx_path=ONNX_PATH, saved_model_dir="models/saved_model"):
    onnx_model = onnx.load(str(onnx_path))
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(saved_model_dir)
    print(f"Saved TF SavedModel to {saved_model_dir}")
    return saved_model_dir


def representative_dataset():
    """Feeds a handful of random samples for int8 calibration.
    Replace with real validation images for better calibration accuracy."""
    for _ in range(100):
        yield [np.random.rand(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)]


def convert_to_tflite(saved_model_dir, int8=True):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    if int8:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        out_path = TFLITE_INT8_PATH
    else:
        out_path = TFLITE_FP32_PATH

    tflite_model = converter.convert()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tflite_model)
    print(f"Saved TFLite model to {out_path} ({len(tflite_model) / 1024:.1f} KB)")


def main():
    model = load_trained_model()
    export_onnx(model)
    saved_model_dir = onnx_to_saved_model()
    convert_to_tflite(saved_model_dir, int8=True)


if __name__ == "__main__":
    main()
