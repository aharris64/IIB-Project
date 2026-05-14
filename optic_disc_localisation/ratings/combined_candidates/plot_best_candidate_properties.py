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

def plot_kde(df, property, name, num_points, cmap):
    x_grid = np.linspace(df[property].min(),
                     df[property].max(), num_points)

    for r in range(1,5):
        vals = df.loc[df["rating"]==r, property].dropna()
        if len(vals) < 2:
            continue

        kde = gaussian_kde(vals)
        plt.plot(x_grid, kde(x_grid), label=f"Rating {r}", c=cmap(r-1))

    plt.xlabel(name)
    plt.ylabel("Density")
    plt.title(f"Keneral Density Estimate (KDE) of {name}")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_kde_binary(df, property, name, num_points, cmap):
    x_grid = np.linspace(df[property].min(),
                         df[property].max(), num_points)

    groups = {
        "Good": df[df["rating"].isin([1, 2])][property].dropna(),
        "Poor": df[df["rating"].isin([3, 4])][property].dropna(),
    }

    for i, (label, vals) in enumerate(groups.items()):
        if len(vals) < 2:
            continue
        kde = gaussian_kde(vals)
        plt.plot(x_grid, kde(x_grid), label=label, c=cmap(i))

    plt.xlabel(name)
    plt.ylabel("Density")
    plt.title(f"Kernel Density Estimate (KDE) of {name}")
    plt.legend()
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


disc_localisation_path = Path(__file__).parents[1]
csv = disc_localisation_path / "all_candidates" / "best_candidates_050226.csv"
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
plot_kde(df, "weighted_score", "Weighted Score", num_points, cmap)
plot_kde_binary(df, "weighted_score", "Weighted Score", num_points, cmap)

def plot_localisation_success(df):
    """
    Stacked 100% bar chart showing localisation rating distribution
    per class, plus an overall bar.
    """
    rating_colors = {1: "#185FA5", 2: "#1D9E75", 3: "#BA7517", 4: "#A32D2D"}
    rating_labels = {1: "1 – Perfect", 2: "2 – Pretty good",
                     3: "3 – Not great", 4: "4 – Completely off"}

    # Add an "Overall" group to the class breakdown
    df = df.copy()
    df_all = df.copy()
    df_all["class"] = "Overall"
    df_combined = pd.concat([df, df_all], ignore_index=True)

    classes = [c for c in sorted(df["class"].unique())] + ["Overall"]

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
        bars = ax.bar(x, pcts, bottom=bottoms,
                      color=rating_colors[r], label=rating_labels[r],
                      width=0.55, edgecolor="white", linewidth=0.8)
        # Annotate each segment with count + %
        for bar, cls, pct in zip(bars, classes, pcts):
            n = summary.loc[cls, f"r{r}_n"]
            if pct > 4:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottoms[classes.index(cls)] + pct / 2,
                    f"{n}\n({pct:.0f}%)",
                    ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold",
                )
        bottoms += pcts

    # Annotate total n above each bar
    for i, cls in enumerate(classes):
        ax.text(i, 102, f"n={summary.loc[cls, 'total']}",
                ha="center", va="bottom", fontsize=8, color="#444441")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Percentage of images (%)", fontsize=10)
    ax.set_title("Disc localisation success by class", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", fontsize=9, frameon=False)

    plt.tight_layout()
    plt.savefig("localisation_success.png", dpi=150, bbox_inches="tight")
    plt.show()

plot_localisation_success(df)