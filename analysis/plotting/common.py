"""Shared setup and loading helpers for the per-run plotting scripts in this package —
factored out since they were previously duplicated across several files."""

import os

import matplotlib.pyplot as plt
import numpy as np

CLASS_NAMES = ["normal", "papilloedema", "pseudo-\npapilloedema"]
NUM_CLASSES = len(CLASS_NAMES)


def setup_style():
    """Apply the consistent font-size styling used across all plots in this package."""
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
    })


def load_predictions(run_root, split):
    """Load y_true/y_pred/y_prob arrays saved by classifier/run_model.ipynb for split."""
    path = os.path.join(run_root, f"predictions_{split}.npz")
    data = np.load(path)
    y_true = data["y_true"].astype(int)
    y_pred = data["y_pred"].astype(int)
    y_prob = data["y_prob"].astype(float)
    return y_true, y_pred, y_prob


def one_hot_encoding(y, num_classes=NUM_CLASSES):
    """One-hot encode integer class labels y into an (N, num_classes) array."""
    one_hot = np.zeros((y.shape[0], num_classes), dtype=int)
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot
