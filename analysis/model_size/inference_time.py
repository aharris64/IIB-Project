import torch
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))
from cnn.models import build_model

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
BATCH_SIZE  = 1
INPUT_SIZE  = (BATCH_SIZE, 3, 224, 224)
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dummy_input = torch.randn(INPUT_SIZE).to(DEVICE)

results = []

for name in MODELS:
    model = build_model(name, NUM_CLASSES, freeze="none").to(DEVICE)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())

    with torch.no_grad():
        # Warmup
        for _ in range(WARMUP):
            _ = model(dummy_input)

        # Time
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(RUNS):
                _ = model(dummy_input)
            end.record()
            torch.cuda.synchronize()
            avg_ms = start.elapsed_time(end) / RUNS
        else:
            t0 = time.perf_counter()
            for _ in range(RUNS):
                _ = model(dummy_input)
            avg_ms = (time.perf_counter() - t0) / RUNS * 1000

    results.append((name, total_params, avg_ms))
    print(f"{name:25s}  params: {total_params:>10,}  avg: {avg_ms:.2f} ms")

# Sorted by inference time
print("\n--- Ranked by speed ---")
for name, params, ms in sorted(results, key=lambda x: x[2]):
    print(f"{name:25s}  {params:>10,} params  {ms:.2f} ms")