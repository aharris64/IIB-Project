"""Reliability-diagram calibration plot: bins predictions by confidence and compares
binned accuracy against confidence, with Wilson-interval error bars per bin."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from plotting.common import setup_style, load_predictions

setup_style()


def create_bins(n_bins, confidence, correct):
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    bin_acc = np.zeros(n_bins, dtype=float)
    bin_conf = np.zeros(n_bins, dtype=float)
    bin_count = np.zeros(n_bins, dtype=int)

    bin_idx = np.digitize(confidence, edges[1:-1], right=True)

    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        bin_count[b] = count
        if count > 0:
            bin_acc[b] = correct[mask].mean()
            bin_conf[b] = confidence[mask].mean()
        else:
            bin_acc[b] = np.nan
            bin_conf[b] = np.nan

    return bin_acc, bin_conf, bin_count, edges

def binomial_ci(k, n, ci = 0.95):
    if n == 0:
        return np.nan, np.nan
    
    z = norm.ppf(1 - (1 - ci) / 2)

    p = k / n
    denom = 1 + (z**2 / n)
    centre = (p + (z**2 / (2 * n)))
    margin = (z / (2 * n)) * np.sqrt(4 * n * p * (1 - p) + z**2)

    lower = (centre - margin) / denom
    upper = (centre + margin) / denom

    return lower, upper

def calculate_error_bars(bin_acc, bin_count):
    bin_lower = np.zeros_like(bin_acc)
    bin_upper = np.zeros_like(bin_acc)

    for i, (accuracy, count) in enumerate(zip(bin_acc, bin_count)):
        if count > 0 and not np.isnan(accuracy):
            k = int(round(accuracy * count))
            low, high = binomial_ci(k, count)
            bin_lower[i] = accuracy - low
            bin_upper[i] = high - accuracy
        else:
            bin_lower[i] = np.nan
            bin_upper[i] = np.nan

    return bin_lower, bin_upper

def plot_calibration_curve(run_root, split, n_bins, save_path=None):

    y_true, y_pred, y_prob = load_predictions(run_root, split)
    confidence = np.max(y_prob, axis=1)
    correct = (y_pred == y_true).astype(int)

    bin_acc, bin_conf, bin_count, edges = create_bins(n_bins, confidence, correct)

    mids = (edges[:-1] + edges[1:]) / 2.0
    widths = (edges[1:] - edges[:-1])

    
    fig, ax = plt.subplots()

    ax.plot([0, 1], [0, 1], linestyle=":", color="black")

    bin_lower, bin_upper = calculate_error_bars(bin_acc, bin_count)
    mask = ~np.isnan(bin_acc) & ~np.isnan(bin_conf)
    ax.errorbar(bin_conf[mask], bin_acc[mask], yerr=[bin_lower[mask], bin_upper[mask]], fmt="o-", capsize=3)

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax2 = ax.twinx()
    ax2.bar(mids, bin_count, width=widths, alpha=0.3, align="center", edgecolor='grey', linewidth=1)
    ax2.set_ylabel("Count")

    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)

    plt.show()