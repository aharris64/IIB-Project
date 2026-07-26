"""Driver script for the plotting library in this package.

List one run name in RUNS to produce the full set of single-run plots (summary,
confusion matrix, loss/F1 curves, PR, ROC, threshold sweep). List several to instead
compare just the loss/F1 curves across runs, one line per run labelled by MODEL_NAME.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plotting.calibration import plot_calibration_curve
from plotting.confusion_matrix import plot_confusion_matrix
from plotting.history import plot_loss, plot_macro_F1
from plotting.precision_recall import plot_pr_indivisual_class, plot_avg_pr
from plotting.roc_auc import plot_roc_ovr, plot_avg_roc
from plotting.threshold import plot_threshold_single_class, plot_threshold_single_metric
from plotting.summary import print_summary

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_FOLDER = REPO_ROOT / "outputs" / "runs"
SAVE_FOLDER = REPO_ROOT / "outputs" / "results"

# RUNS = ["efficientnet_b0_20260520_161651",
#         "efficientnet_lite0_20260520_164934",
#         "ghostnet_20260520_181408",
#         "mobilenet_v3_20260520_121742",
#         "mobilenet_v3_small_20260520_141418",
#         "resnet_20260520_154113",
#         "squeezenet_20260520_174646"]

# w/o l0
# RUNS = ["efficientnet_b0_20260520_161651",
#         "ghostnet_20260520_181408",
#         "mobilenet_v3_20260520_121742",
#         "mobilenet_v3_small_20260520_141418",
#         "resnet_20260520_154113",
#         "squeezenet_20260520_174646"]

RUNS = ["mobilenet_v3_small_20260520_185910"]

split = "test"

run_paths = [os.path.join(RUN_FOLDER, r) for r in RUNS]

if len(run_paths) == 1:
    run_path = run_paths[0]
    save_path = os.path.join(SAVE_FOLDER, RUNS[0])

    print_summary(run_path, save_path)

    n_bins = 10
    # plot_calibration_curve(run_path, split, n_bins)

    plot_confusion_matrix(run_path, split, False)
    plot_confusion_matrix(run_path, split, True)

    plot_loss(run_path)
    plot_macro_F1(run_path)

    plot_pr_indivisual_class(run_path, split)
    plot_avg_pr(run_path, split)

    plot_roc_ovr(run_path, split)
    plot_avg_roc(run_path, split)

    # plot_threshold_single_class(run_path, split, 0)
    # plot_threshold_single_class(run_path, split, 1)
    # plot_threshold_single_class(run_path, split, 2)

    # plot_threshold_single_metric(run_path, split, "precision")
    # plot_threshold_single_metric(run_path, split, "recall")
    # plot_threshold_single_metric(run_path, split, "f1")

else:
    plot_loss(run_paths, train_loss=True, val_loss=False)
    plot_loss(run_paths, train_loss=False, val_loss=True)
    plot_macro_F1(run_paths)
