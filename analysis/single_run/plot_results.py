import os

from calibration import plot_calibration_curve
from confusion_matrix import plot_confusion_matrix
from history import plot_loss, plot_macro_F1
from precision_recall import plot_pr_indivisual_class, plot_avg_pr
from roc_auc import plot_roc_ovr, plot_avg_roc
from threshold import plot_threshold_single_class, plot_threshold_single_metric
from summary import print_summary

# run = "efficientnet_b0_20260520_161651"
# run = "efficientnet_lite0_20260520_164934"
# run = "ghostnet_20260520_181408"
# run = "mobilenet_v3_20260520_121742"
# run = "mobilenet_v3_small_20260520_141418"
# run = "resnet_20260520_154113"
# run = "squeezenet_20260520_174646"

# run = "mobilenet_v3_20260520_210306"
run = "mobilenet_v3_small_20260520_185910"

run_folder = r"C:\Users\adam6\OneDrive\Documents\University\Engineering\Engineering IIB\IIB Project\Code\IIB-Project\runs"
save_folder =  r"C:\Users\adam6\OneDrive\Documents\University\Engineering\Engineering IIB\IIB Project\Code\IIB-Project\results"

run_path = os.path.join(run_folder, run)
save_path = os.path.join(save_folder, run)

split = "test"

print_summary(run_path, save_path)

n_bins = 10
# plot_calibration_curve(run_path, split, n_bins)

# normalised = False
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