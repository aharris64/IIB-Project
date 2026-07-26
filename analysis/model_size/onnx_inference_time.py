"""Exports each architecture to ONNX (into outputs/onnx_models/) and benchmarks its
onnxruntime CPU inference time and on-disk size — contrast with inference_time.py,
which times raw PyTorch (no export) instead."""

import torch
import onnx
import onnxruntime as ort
import numpy as np
import time
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[2]))
from classifier.models import build_model

MODELS = [
    "efficientnet_b0",
    "mobilenet_v3",
    "mobilenet_v3_small",
    "squeezenet",
    "resnet",
    "efficientnet_lite0",
    "ghostnet",
]

NUM_CLASSES = 3
WARMUP      = 20
RUNS        = 100
INPUT_SIZE  = (1, 3, 224, 224)

dummy_input   = torch.randn(INPUT_SIZE)
dummy_np      = dummy_input.numpy()

ONNX_MODELS_DIR = Path(__file__).resolve().parents[2] / "outputs" / "onnx_models"
os.makedirs(ONNX_MODELS_DIR, exist_ok=True)
results = []

for name in MODELS:
    model = build_model(name, NUM_CLASSES, freeze="none")
    model.eval()

    # Export to ONNX
    onnx_path = str(ONNX_MODELS_DIR / f"{name}.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
        export_params=True
    )

    # Run with onnxruntime
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    # Warmup
    for _ in range(WARMUP):
        sess.run(None, {input_name: dummy_np})

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(RUNS):
        sess.run(None, {input_name: dummy_np})
    avg_ms = (time.perf_counter() - t0) / RUNS * 1000

    # File size
    size_mb = os.path.getsize(onnx_path + ".data") /1e6

    results.append((name, avg_ms, size_mb))
    print(f"{name:25s}  {avg_ms:.2f} ms  {size_mb:.1f} MB")

print("\n--- Ranked by speed ---")
for name, ms, mb in sorted(results, key=lambda x: x[1]):
    print(f"{name:25s}  {ms:.2f} ms  {mb:.1f} MB")