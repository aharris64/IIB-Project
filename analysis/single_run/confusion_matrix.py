import json
import matplotlib.pyplot as plt
import numpy as np
import os

class_names = ["normal", "papilloedema", "pseudo-\npapilloedema"]
n = len(class_names)

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

def load_confusion_matrix(run_root, split, normalised):
    """
    Loads confusion matrix from json file 'metrics' given a folder for the run
    Returns a confusion matrix, normalised if True
    """
    metrics_file = os.path.join(run_root, "metrics.json")
    with open(metrics_file) as f:
        metrics = json.load(f)

    confusion_matrix = np.array(metrics[split]["confusion_matrix"], dtype=int)
    
    if normalised:
        confusion_matrix = confusion_matrix.astype(float)
        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        confusion_matrix = np.divide(confusion_matrix, row_sums, where=row_sums != 0)

    return confusion_matrix

def plot_confusion_matrix(run_root, split, normalised, save_path=None):
    """
    Plots confusion matrix with colourmap
    """

    confusion_matrix = load_confusion_matrix(run_root, split, normalised)

    for i in range(n):
        for j in range(n):
            if normalised:
                txt = f"{confusion_matrix[i, j]:.4f}"
            else:
                txt = str(int(confusion_matrix[i, j]))
            plt.text(
                j, i, txt,
                ha="center", va="center",
                color="white" if confusion_matrix[i, j] > confusion_matrix.max() / 2 else "black"
            )

    plt.imshow(confusion_matrix, cmap="Blues")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    plt.xticks(np.arange(n), class_names)
    plt.yticks(np.arange(n), class_names)

    plt.colorbar()
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)

    plt.show()