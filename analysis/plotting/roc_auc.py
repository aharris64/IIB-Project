"""ROC curves: per-class (one-vs-rest, with AUC) and micro-averaged across all classes."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score

from plotting.common import setup_style, CLASS_NAMES, NUM_CLASSES, load_predictions, one_hot_encoding

setup_style()


def plot_roc_ovr(run_root, split, save_path=None):
    y_true, _, y_prob = load_predictions(run_root, split)
    y_true_oh = one_hot_encoding(y_true)

    per_class_auc = {}
    for c in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_true_oh[:, c], y_prob[:, c])
        roc_auc = auc(fpr, tpr)
        per_class_auc[c] = roc_auc
        plt.plot(fpr, tpr, label=f"{CLASS_NAMES[c]} (AUC={roc_auc:.3f})")

    macro_auc = np.mean(list(per_class_auc.values()))
    print(f"Macro-AUC: {macro_auc:.4f}")

    plt.plot([0, 1], [0, 1], linestyle=":", color="black") # Chance line

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)

    plt.show()

def plot_avg_roc(run_root, split, save_path=None):
    y_true, _, y_prob = load_predictions(run_root, split)
    y_true_oh = one_hot_encoding(y_true)

    fpr, tpr, _ = roc_curve(y_true_oh.ravel(), y_prob.ravel())
    auc_micro_score = roc_auc_score(y_true_oh, y_prob, average="micro", multi_class="ovr")

    plt.plot(fpr, tpr, label=f"Micro-AUC={auc_micro_score:.3f}")

    # Chance line
    plt.plot([0, 1], [0, 1], linestyle=":", color="black")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)

    plt.show()