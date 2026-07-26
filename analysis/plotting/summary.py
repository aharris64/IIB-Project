"""Prints (and optionally saves) a plain-text summary of one run: identity, config,
best epoch, and val/test metrics — pulled from that run's config/run_meta/metrics JSON."""

import json
import os
from pathlib import Path

def load_json(path):
    with open(path) as f:
        return json.load(f)

def print_summary(run_dir, save_path = None):

    cfg = load_json(os.path.join(run_dir, "config.json"))
    meta = load_json(os.path.join(run_dir, "run_meta.json"))
    metrics = load_json(os.path.join(run_dir, "metrics.json"))

    lines = []

    # --- Run identity ---
    lines.append(f"Run ID: {meta['run_id']}")
    lines.append(f"Timestamp: {meta['timestamp']}")
    lines.append(f"Device: {meta['device']}")
    lines.append(f"Seed: {meta['seed']}")
    lines.append(f"Note: {cfg['EXPERIMENT_NOTE']}")
    lines.append("")

    # --- Model / training ---
    lines.append("Model configuration:")
    lines.append(f"  Model: {cfg['MODEL_NAME']}")
    lines.append(f"  Dataset: {cfg['DATASET']}")
    lines.append(f"  Num classes: {cfg['NUM_CLASSES']}")
    lines.append(f"  Freeze: {cfg['FREEZE']}")
    lines.append(f"  Batch size: {cfg['BATCH_SIZE']}")
    lines.append(f"  Learning rate: {cfg['LEARNING_RATE']}")
    lines.append(f"  Weight decay: {cfg['WEIGHT_DECAY']}")
    lines.append(f"  Epochs: {cfg['NUM_EPOCHS']}")
    lines.append(f"  Patience: {cfg['PATIENCE']}")
    lines.append("")

    # --- Model selection ---
    lines.append(f"Best epoch: {metrics['best_epoch']}")
    lines.append("")

    # --- Validation metrics ---
    val = metrics["val"]
    lines.append("Validation:")
    lines.append(f"  Loss: {val['loss']:.4f}")
    lines.append(f"  Macro F1: {val['macro_f1']:.4f}")
    lines.append(f"  Balanced Acc: {val['balanced_acc']:.4f}")
    lines.append("")

    # --- Test metrics ---
    test = metrics["test"]
    lines.append("Test:")
    lines.append(f"  Loss: {test['loss']:.4f}")
    lines.append(f"  Macro F1: {test['macro_f1']:.4f}")
    lines.append(f"  Balanced Acc: {test['balanced_acc']:.4f}")

    summary_text = "\n".join(lines)

    # Print to console
    print(summary_text)

    # Optionally save
    if save_path is not None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        out_path = save_path / "summary.txt"
        with out_path.open("w") as f:
            f.write(summary_text)

