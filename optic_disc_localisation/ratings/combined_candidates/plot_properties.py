import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_auc_score

def calculate_wegihted_score(df):
    # Define weights
    w_contrast = 3.831443
    w_response = -8.454572
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

    legend = ["Very Good", "Good", "Poor", "Very Poor"]

    for r in range(1,5):
        vals = df.loc[df["rating"]==r, property].dropna()
        if len(vals) < 2:
            continue

        kde = gaussian_kde(vals)
        weight = len(vals) / len(df[property].dropna())  # fraction of total
        plt.plot(x_grid, kde(x_grid) * weight, label=legend[r-1], c=cmap(r-1))

    

    plt.xlabel(name, fontsize=12)
    plt.ylabel("Density",fontsize=12)
    plt.gca().spines[["top", "right"]].set_visible(False)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
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


disc_localisation_path = Path(__file__).parents[1]
csv = disc_localisation_path / "combined_candidates" / "all_candidates_050226.csv"
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

plot_kde(df, "weighted_score", "Weighted Score", num_points, cmap)