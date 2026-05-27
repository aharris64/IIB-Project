import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score, average_precision_score
import os

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

class_names = ["normal", "papilloedema", "pseudo-\npapilloedema"]
n = 3

def load_predictions(run_root, split):
    path = os.path.join(run_root, f"predictions_{split}.npz")
    data = np.load(path)
    y_true = data["y_true"].astype(int)
    y_prob = data["y_prob"].astype(float) 
    return y_true, y_prob

def one_hot_encoding(y):
    one_hot = np.zeros((y.shape[0], n), dtype=int)
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot

def plot_roc_ovr(run_root, split, save_path=None):
    y_true, y_prob = load_predictions(run_root, split)
    y_true_oh = one_hot_encoding(y_true)

    per_class_auc = {}
    for c in range(n):
        fpr, tpr, _ = roc_curve(y_true_oh[:, c], y_prob[:, c])
        roc_auc = auc(fpr, tpr)
        per_class_auc[c] = roc_auc
        plt.plot(fpr, tpr, label=f"{class_names[c]} (AUC={roc_auc:.3f})")

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
    y_true, y_prob = load_predictions(run_root, split)
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