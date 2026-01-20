import os

from history import plot_loss, plot_macro_F1

runs = ["efficientnet_b0_20260111_214118",
        "efficientnet_lite0_20260114_152301",
        "ghostnet_20260114_154543",
        "mobilenet_v2_20260114_125208",
        "resnet_20260114_151502",
        "squeezenet_20260114_150400"]

# Without efficient net lite 0
runs = ["efficientnet_b0_20260111_214118",
        "ghostnet_20260114_154543",
        "mobilenet_v2_20260114_125208",
        "resnet_20260114_151502",
        "squeezenet_20260114_150400"]
        

run_folder = r"C:\Users\adam6\OneDrive\Documents\University\Engineering\Engineering IIB\IIB Project\Code\IIB-Project\runs"

runs_root = []
for run in runs:
    run_path = os.path.join(run_folder, run)
    runs_root.append(run_path)

split = "test"

# print_summary(run_path, save_path)

# n_bins = 10
# plot_calibration_curve(run_path, split, n_bins)

# normalised = False
# plot_confusion_matrix(run_path, split, False)
# plot_confusion_matrix(run_path, split, True)

plot_loss(runs_root, train_loss=True, val_loss=False)
plot_loss(runs_root, train_loss=False, val_loss=True)
plot_macro_F1(runs_root)

# plot_pr_indivisual_class(run_path, split)
# plot_avg_pr(run_path, split)

# plot_roc_ovr(run_path, split)
# plot_avg_roc(run_path, split)

# plot_threshold_single_class(run_path, split, 0)
# plot_threshold_single_class(run_path, split, 1)
# plot_threshold_single_class(run_path, split, 2)

# plot_threshold_single_metric(run_path, split, "precision")
# plot_threshold_single_metric(run_path, split, "recall")
# plot_threshold_single_metric(run_path, split, "f1")