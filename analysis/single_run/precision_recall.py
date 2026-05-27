import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
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

def plot_pr_indivisual_class(run_root, split, save_path=None):
    y_true, y_prob = load_predictions(run_root, split)

    ap_per_class = []
    for c in range(n):
        y_true_bin = (y_true == c).astype(int)
        y_score = y_prob[:, c]

        precision, recall, _ = precision_recall_curve(y_true_bin, y_score)
        ap = average_precision_score(y_true_bin, y_score)

        ap_per_class.append(ap)
        plt.plot(recall, precision, label=f"{class_names[c]} (AP={ap:.3f})")
    
    macro_ap = np.mean(ap_per_class)
    print(f"Macro-AP: {macro_ap:.4f}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)

    plt.show()

def plot_avg_pr(run_root, split, save_path=None):
    y_true, y_prob = load_predictions(run_root, split)
    y_true_oh = one_hot_encoding(y_true)

    precision, recall, _ = precision_recall_curve(y_true_oh.ravel(),y_prob.ravel())
    ap_micro = average_precision_score(y_true_oh, y_prob, average="micro")

    plt.plot(recall, precision, label=f"Micro-AP={ap_micro:.3f}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)

    plt.show()