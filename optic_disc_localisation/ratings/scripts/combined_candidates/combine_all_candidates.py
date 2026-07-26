"""Stacks the "other candidates" and "best candidate" rating sets into one table for
learning (used by learn_weights.py). NOTE: paths below were previously broken
(referenced folders like "candidate_ratings"/"all_candidates_for_learning" that don't
exist anywhere in the repo) — fixed to point at the real data/ locations, matching
each source file's exact real name."""

import pandas as pd
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
other_candidates = DATA_ROOT / "other_candidate_ratings" / "disc_candidates_050226_combined.csv"
best_candidates    = DATA_ROOT / "best_candidate_ratings_trial_weights" / "disc_localisation_030226_combined.csv"
outpath       = DATA_ROOT / "combined_candidates" / "all_candidates_050226.csv"

df_other = pd.read_csv(other_candidates)
df_best = pd.read_csv(best_candidates)

# normalize image to "normal_0005_EDD" for both
def clean_image(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace(r"^.*/", "", regex=True)   # drop folders
    s = s.str.replace(r"\.[^.]+$", "", regex=True)           # drop any extension
    return s

df_other["image"] = clean_image(df_other["image"])
df_best["image"] = clean_image(df_best["image"])

# current best has candidate_num = 0
df_best["candidate_num"] = 0

# class = class_x
df_best = df_best.rename(columns={"class_x": "class"})
df_other = df_other.rename(columns={"class_x": "class"})

# output columns
out_cols = [
    "image",
    "candidate_num",
    "rating",
    "class",
    "centre",
    "radius",
    "blob_score",
    "blob_contrast",
    "blob_brightness",
    "blob_response",
    "final_score",
    "vessel_centre",
]


# --- stack them ---
out = pd.concat([df_other[out_cols], df_best[out_cols]], ignore_index=True)

# sort so candidate_num=0 comes first per image
out = out.sort_values(["image", "candidate_num"], kind="stable").reset_index(drop=True)

out.to_csv(outpath, index=False)
print("Wrote final_combined.csv with", len(out), "rows")
