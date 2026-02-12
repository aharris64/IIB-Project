import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_auc_score

def calculate_wegihted_score(df):
    # Define weights
    w_contrast = 3.83144
    w_response = -8.45457
    w_final    = 1.21192
    bias       = 0.
    
    df["final_sign"] = (df["final_score"] > 0).astype(int) # Just use sign

    # Compute score
    df["weighted_score"] = (
        w_contrast * df["blob_contrast"]
        + w_response * df["blob_response"]
        + w_final * df["final_sign"]
        + bias
    )

    return df

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.optimize import brentq
from itertools import combinations

def kde_objects_by_rating(df, property, ratings=(1,2,3,4)):
    kdes = {}
    for r in ratings:
        vals = df.loc[df["rating"] == r, property].dropna().to_numpy()
        if len(vals) >= 2:
            kdes[r] = gaussian_kde(vals)
    return kdes

def find_all_crossings(kdes, x_grid, eps=1e-12):
    """
    kdes: dict {rating: gaussian_kde}
    x_grid: common grid (1D array)
    Returns: dict {(r1,r2): [x_crossings...]} sorted
    """
    crossings = {}

    # pre-evaluate each kde on the grid for robust bracketing
    Y = {r: kdes[r](x_grid) for r in kdes.keys()}

    for r1, r2 in combinations(sorted(kdes.keys()), 2):
        d = Y[r1] - Y[r2]

        # indices where sign changes between consecutive points
        s = np.sign(d)
        s[np.abs(d) < eps] = 0.0

        idx = np.where(s[:-1] * s[1:] < 0)[0]  # strict sign change
        xs = []

        def f(x):
            return kdes[r1](x)[0] - kdes[r2](x)[0]

        for i in idx:
            a, b = x_grid[i], x_grid[i+1]
            # brentq needs opposite signs; guard against numerical issues
            fa, fb = f(a), f(b)
            if fa == 0:
                xs.append(a)
                continue
            if fb == 0:
                xs.append(b)
                continue
            if fa * fb > 0:
                continue
            try:
                root = brentq(f, a, b, maxiter=200)
                xs.append(root)
            except ValueError:
                pass

        # de-duplicate very close roots
        xs = sorted(xs)
        xs_unique = []
        for x in xs:
            if not xs_unique or abs(x - xs_unique[-1]) > (x_grid[1] - x_grid[0]) * 2:
                xs_unique.append(x)

        crossings[(r1, r2)] = xs_unique

    return crossings

def plot_kde_with_crossings(df, property, name, num_points, cmap, mark_crossings=True):
    # common grid across all data (not per-rating)
    x_min, x_max = df[property].min(), df[property].max()
    x_grid = np.linspace(x_min, x_max, num_points)

    kdes = kde_objects_by_rating(df, property, ratings=(1,2,3,4))

    # plot curves
    for r in range(1, 5):
        if r not in kdes:
            continue
        plt.plot(x_grid, kdes[r](x_grid), label=f"Rating {r}", c=cmap(r-1))

    # crossings
    crossings = find_all_crossings(kdes, x_grid)

    if mark_crossings:
        for (r1, r2), xs in crossings.items():
            for x in xs:
                y = kdes[r1](x)[0]  # equals kdes[r2](x)
                plt.scatter([x], [y], s=25, marker="x", c="k")

    plt.xlabel(name)
    plt.ylabel("Density")
    plt.title(f"Kernel Density Estimate (KDE) of {name}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # print crossings
    for pair, xs in crossings.items():
        r1, r2 = pair
        print(f"Crossings Rating {r1} vs {r2}:")
        if xs:
            for x in xs:
                print(f"  x = {x:.6f}")
        else:
            print("  (none)")
    return crossings


def count_ratings(df):
    counts = df["rating"].value_counts().sort_index()
    total = len(df)

    print("Rating distribution:\n")

    for r in [1, 2, 3, 4]:
        n = counts.get(r, 0)
        pct = 100 * n / total if total > 0 else 0

        print(f"Rating {r}: {n:5d} samples  ({pct:6.2f}%)")

    print(f"\nTotal samples: {total}")

def threshold_loss_report(df, threshold, mode="below"):
    """
    mode:
        "below"  -> lost = weighted_score < threshold
        "above"  -> lost = weighted_score > threshold
    """

    print(f"\nThreshold = {threshold:.6f}")
    print(f"Mode      = losing values {mode} threshold\n")

    total_samples = len(df)

    total_lost = 0

    for r in [1, 2, 3, 4]:
        subset = df[df["rating"] == r]
        n_total = len(subset)

        if mode == "below":
            n_lost = (subset["weighted_score"] < threshold).sum()
        elif mode == "above":
            n_lost = (subset["weighted_score"] > threshold).sum()
        else:
            raise ValueError("mode must be 'below' or 'above'")
        
        total_lost += n_lost

        pct_lost = 100 * n_lost / n_total if n_total > 0 else 0

        print(f"Rating {r}:")
        print(f"  Total : {n_total:5d}")
        print(f"  Lost  : {n_lost:5d}  ({pct_lost:6.2f}%)\n")

    total_pct_lost = 100 * total_lost / total_samples
    print(f"All Images:")
    print(f"  Total : {total_samples:5d}")
    print(f"  Lost  : {total_lost:5d}  ({total_pct_lost:6.2f}%)\n")
    print(f"  New Size  : {(total_samples-total_lost):5d}")
    





disc_localisation_path = Path(__file__).parents[1]
csv = disc_localisation_path / "all_candidates" / "best_candidates_050226.csv"
df = pd.read_csv(csv)

count_ratings(df)

df = calculate_wegihted_score(df)

num_points = 1000

cmap = plt.colormaps.get_cmap("bwr").resampled(4)

crossings = plot_kde_with_crossings(df, "weighted_score", "Weighted Score", num_points, cmap)

threshold = 1.5
threshold_loss_report(df, threshold, mode="below")
