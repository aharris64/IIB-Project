"""Precision/recall/F1 vs. decision threshold, per class or one metric across all
classes — useful for picking a non-default classification threshold."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

from plotting.common import setup_style, CLASS_NAMES, NUM_CLASSES, load_predictions

setup_style()

thresholds = np.linspace(0.0, 1.0, 101)


def get_precision_recall_f1(y_true, y_prob, class_idx, thresholds):
    
    y_true_bin = (y_true == class_idx).astype(int)
    scores = y_prob[:, class_idx]

    precision, recall, f1 = [], [], []

    for t in thresholds:
        y_pred_bin = (scores >= t).astype(int)
        precision.append(precision_score(y_true_bin, y_pred_bin, zero_division=0))
        recall.append(recall_score(y_true_bin, y_pred_bin, zero_division=0))
        f1.append(f1_score(y_true_bin, y_pred_bin, zero_division=0))

    return np.array(precision), np.array(recall), np.array(f1)

def plot_threshold_single_class(run_root, split, class_idx, save_path=None):

    y_true, _, y_prob = load_predictions(run_root, split)
    precision, recall, f1 = get_precision_recall_f1(y_true, y_prob, class_idx, thresholds)

    plt.plot(thresholds, precision, label=f"Class {CLASS_NAMES[class_idx]}: Precision")
    plt.plot(thresholds, recall, label=f"Class {CLASS_NAMES[class_idx]}: Recall")
    plt.plot(thresholds, f1, label=f"Class {CLASS_NAMES[class_idx]}: F1")

    idx = np.argmax(f1)
    print(f"Best F1 at threshold for Class {CLASS_NAMES[class_idx]}={thresholds[idx]:.2f}: F1={f1[idx]:.3f}")

    plt.xlabel("Decision Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)

    plt.show()

def plot_threshold_single_metric(run_root, split, metric, save_path=None):
    y_true, _, y_prob = load_predictions(run_root, split)

    for c in range(NUM_CLASSES):
        precision, recall, f1 = get_precision_recall_f1(y_true, y_prob, c, thresholds)
        if metric == "precision":
            plt.plot(thresholds, precision, label=f"Class {CLASS_NAMES[c]}: Precision")
        if metric == "recall":
            plt.plot(thresholds, recall, label=f"Class {CLASS_NAMES[c]}: Recall")
        if metric == "f1":
            plt.plot(thresholds, f1, label=f"Class {CLASS_NAMES[c]}: F1")
            idx = np.argmax(f1)
            print(f"Best F1 at threshold for Class {CLASS_NAMES[c]}={thresholds[idx]:.2f}: F1={f1[idx]:.3f}")

    plt.xlabel("Decision Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)

    plt.show()