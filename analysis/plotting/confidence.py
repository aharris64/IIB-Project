"""Per-image confidence bar chart for a run's test set: one bar per image, height is
the model's confidence (max predicted probability), coloured by whether the
prediction was correct or misclassified. Bars are sorted by confidence so the
split between confident-correct and confident-wrong predictions is visible."""

import numpy as np
import matplotlib.pyplot as plt

from plotting.common import setup_style, load_predictions

setup_style()

CORRECT_COLOR = "#0ca30c"
MISCLASSIFIED_COLOR = "#d03b3b"


def plot_confidence_bar_chart(run_root, split, sort=True, save_path=None):
    y_true, y_pred, y_prob = load_predictions(run_root, split)
    confidence = np.max(y_prob, axis=1)
    correct = y_pred == y_true

    if sort:
        order = np.argsort(confidence)
        confidence = confidence[order]
        correct = correct[order]

    colors = np.where(correct, CORRECT_COLOR, MISCLASSIFIED_COLOR)

    fig, ax = plt.subplots()
    ax.bar(np.arange(len(confidence)), confidence, width=1.0, color=colors)

    ax.set_xlabel("Image (sorted by confidence)" if sort else "Image")
    ax.set_ylabel("Confidence")
    ax.set_xlim(0, len(confidence))
    ax.set_ylim(0, 1)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=CORRECT_COLOR, label="Correct"),
        plt.Rectangle((0, 0), 1, 1, color=MISCLASSIFIED_COLOR, label="Misclassified"),
    ]
    ax.legend(handles=handles)

    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)

    plt.show()
