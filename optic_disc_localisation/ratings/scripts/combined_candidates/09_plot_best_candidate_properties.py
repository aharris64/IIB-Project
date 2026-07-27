"""Step 9 (terminal/standalone, final analysis): KDE plots of weighted score by
rating, an F1-optimal decision threshold, and stacked bar charts of
localisation/vessel-detection success by class — on the reclassified
(05_reclassify_labels.py output) combined-candidates data."""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score
import matplotlib.gridspec as gridspec
import math

def calculate_wegihted_score(df):
    # Define weights
    w_contrast = 3.83144
    w_response = -8.45457
    w_final    = 1.211922
    bias       = 0.5929768734602114
    
    df["final_sign"] = (df["final_score"] > 0).astype(int) # Just use sign

    # Compute score
    df["weighted_score"] = (
        w_contrast * df["blob_contrast"]
        + w_response * df["blob_response"]
        + w_final * df["final_sign"]
        + bias
    )

    return df

def plot_overlapping_histograms(df, property, name, num_bins):

    for r in [1, 2, 3, 4]:
        vals = df.loc[df["rating"] == r, property]
        if len(vals) == 0:
            continue
        plt.hist(vals, bins=num_bins, alpha=0.5, density=True, label=f"rating {r}")

    plt.xlabel(f"{name}")
    plt.ylabel("Density")
    plt.title(f"Overlayed histograms of {name}")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_kde(df, property, name, num_points, cmap, threshold=None):
    x_grid = np.linspace(df[property].min(),
                     df[property].max(), num_points)
    
    legend = ["Very Good", "Good", "Poor", "Very Poor"]

    for r in range(1,5):
        vals = df.loc[df["rating"]==r, property].dropna()
        if len(vals) < 2:
            continue

        kde = gaussian_kde(vals)
        weight = len(vals) / len(df[property].dropna())  # fraction of total
        plt.plot(x_grid, kde(x_grid) * weight, label=legend[r-1], c=cmap(r-1))

    if threshold is not None:
        plt.axvline(threshold, color="black", linestyle="--", label=f"Threshold ({threshold:.2f})")

    plt.xlabel(name, fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.gca().spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.show()

def count_ratings(df):
    counts = df["rating"].value_counts().sort_index()
    total = len(df)

    print("Rating distribution:\n")

    for r in [1, 2, 3, 4]:
        n = counts.get(r, 0)
        pct = 100 * n / total if total > 0 else 0

        print(f"Rating {r}: {n:5d} samples  ({pct:6.2f}%)")

    print(f"\nTotal samples: {total}")


def find_threshold(df, good_ratings=(1, 2)):
    y_true  = df["rating"].isin(good_ratings).astype(int)
    y_score = df["weighted_score"]
    fpr, tpr, roc_thresh = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    precision, recall, pr_thresh = precision_recall_curve(y_true, y_score)

    thresholds = np.arange(math.floor(y_score.min() * 100) / 100, y_score.max(), 0.01)

    f1_scores = [
        f1_score((y_score >= t).astype(int), y_true, average="macro", zero_division=0)
        for t in thresholds
    ]

    best_idx       = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    max_f1         = f1_scores[best_idx]
    y_pred         = (y_score >= best_threshold).astype(int)
    accuracy       = (y_pred == y_true).mean() * 100

    good_mask = y_true == 1
    poor_mask = y_true == 0
    good_accuracy = (y_pred[good_mask] == y_true[good_mask]).mean() * 100
    poor_accuracy = (y_pred[poor_mask] == y_true[poor_mask]).mean() * 100

    print(f"Best Threshold: {best_threshold:.4f}")
    print(f"Max F1 Score:   {max_f1:.4f}")
    print(f"Accuracy:       {accuracy:.2f}%")
    print(f"  Good:         {good_accuracy:.2f}%")
    print(f"  Poor:         {poor_accuracy:.2f}%")
    for r in range(1, 5):
        mask = df["rating"] == r
        if mask.sum() == 0:
            continue
        r_accuracy = (y_pred[mask] == y_true[mask]).mean() * 100
        print(f"  Rating {r}:     {r_accuracy:.2f}%")


    return best_threshold

CLASS_ORDER  = ["normal", "papilledema", "pseudopapilledema", "non_specific_disc_oedema"]
CLASS_LABELS = {
    "normal":                   "Normal",
    "papilledema":              "Papilloedema",
    "pseudopapilledema":        "Pseudo-\npapilloedema",
    "non_specific_disc_oedema": "Non Specific\nDisc Oedema",
}
cmap1 = plt.colormaps.get_cmap("bwr").resampled(6)


def plot_localisation_success(df):
    """
    Stacked 100% bar chart showing localisation rating distribution
    per class, plus an overall bar.
    """
    rating_colors = {1: cmap1(0), 2: cmap1(1), 3: cmap1(4), 4: cmap1(5)}
    rating_labels = {1: "1 Very Good", 2: "2 Good", 3: "3 Poor", 4: "4 Very Poor"}

    # Add an "Overall" group to the class breakdown
    df = df.copy()
    df_all = df.copy()
    df_all["class"] = "Overall"
    df_combined = pd.concat([df, df_all], ignore_index=True)

    classes = [c for c in CLASS_ORDER if c in df["class"].unique()] + ["Overall"]
    class_labels = [CLASS_LABELS[c] for c in classes[:-1]] + ["Overall"]

    # Build percentage table
    rows = []
    for cls in classes:
        sub = df_combined[df_combined["class"] == cls]
        total = len(sub)
        row = {"class": cls, "total": total}
        for r in [1, 2, 3, 4]:
            n = (sub["rating"] == r).sum()
            row[f"r{r}_n"]   = n
            row[f"r{r}_pct"] = 100 * n / total if total > 0 else 0
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("class").loc[classes]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")

    x = np.arange(len(classes))
    bottoms = np.zeros(len(classes))

    for r in [1, 2, 3, 4]:
        pcts = summary[f"r{r}_pct"].values
        bars = ax.bar(x, pcts, bottom=bottoms, color=rating_colors[r], label=rating_labels[r])
        # Annotate each segment with count + %
        for bar, cls, pct in zip(bars, classes, pcts):
            n = summary.loc[cls, f"r{r}_n"]
            if pct > 4:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottoms[classes.index(cls)] + pct / 2,
                    f"{n} ({pct:.0f}%)",
                    ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold",
                )
        bottoms += pcts

    # Annotate total n above each bar
    for i, cls in enumerate(classes):
        ax.text(i, 102, f"n={summary.loc[cls, 'total']}",
                ha="center", va="bottom", fontsize=9, color="k")

    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, fontsize=12)
    ax.set_ylabel("Percentage of Images (%)", fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncols=4, fontsize=11, frameon=True)

    plt.tight_layout()
    plt.show()

def plot_vessel_ok(df):
    """
    Stacked 100% bar chart showing vessel detection success by class, plus an overall bar.
    """
    vessel_colors = {9: cmap1(0), 0: cmap1(5)}
    vessel_labels = {9: "Vessel Centre Correct", 0: "Vessel Centre Incorrect"}
 
    # Add an "Overall" group
    df = df.copy()
    df_all = df.copy()
    df_all["class"] = "Overall"
    df_combined = pd.concat([df, df_all], ignore_index=True)
 
    classes = [c for c in CLASS_ORDER if c in df["class"].unique()] + ["Overall"]
    class_labels = [CLASS_LABELS[c] for c in classes[:-1]] + ["Overall"]
 
    # Build percentage table
    rows = []
    for cls in classes:
        sub = df_combined[df_combined["class"] == cls]
        total = len(sub)
        row = {"class": cls, "total": total}
        for v in [9, 0]:
            n = (sub["vessel_ok"] == v).sum()
            row[f"v{v}_n"]   = n
            row[f"v{v}_pct"] = 100 * n / total if total > 0 else 0
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("class").loc[classes]
 
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
 
    x = np.arange(len(classes))
    bottoms = np.zeros(len(classes))
 
    for v in [9, 0]:
        pcts = summary[f"v{v}_pct"].values
        bars = ax.bar(x, pcts, bottom=bottoms, color=vessel_colors[v], label=vessel_labels[v])
        for bar, cls, pct in zip(bars, classes, pcts):
            n = summary.loc[cls, f"v{v}_n"]
            if pct > 4:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottoms[classes.index(cls)] + pct / 2,
                    f"{n} ({pct:.0f}%)",
                    ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold",
                )
        bottoms += pcts
 
    # Annotate total n above each bar
    for i, cls in enumerate(classes):
        ax.text(i, 102, f"n={summary.loc[cls, 'total']}",
                ha="center", va="bottom", fontsize=9, color="k")
 
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, fontsize=12)
    ax.set_ylabel("Percentage of Images (%)", fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncols=2, fontsize=11, frameon=True)
 
    plt.tight_layout()
    plt.show()
 

disc_localisation_path = Path(__file__).resolve().parents[2] / "data"
csv = disc_localisation_path / "combined_candidates" / "best_candidates_reclassified.csv"
df = pd.read_csv(csv)

count_ratings(df)

df = calculate_wegihted_score(df)


bins = 100
num_points = 1000

cmap = plt.colormaps.get_cmap("bwr").resampled(4)

# plot_overlapping_histograms(df, "blob_contrast", "Contrast", bins )
# plot_overlapping_histograms(df, "blob_response", "DoG Response", bins)
# plot_overlapping_histograms(df, "final_score", "Vessel Score", bins)

# plot_kde(df, "blob_contrast", "Contrast", num_points, cmap)
# plot_kde(df, "blob_response", "DoG Response", num_points, cmap)
# plot_kde(df, "final_score", "Vessel Score", num_points, cmap)

# plot_kde(df, "final_score", "Vessel Score", num_points, cmap)


threshold = find_threshold(df)
plot_kde(df, "weighted_score", "Weighted Score", num_points, cmap, threshold)

plot_localisation_success(df)
 
plot_vessel_ok(df)