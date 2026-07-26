"""Picks the max-blob_score candidate per image. NOTE: paths below were previously
broken (phantom "all_candidates" folder) — fixed to the real combined_candidates/
location."""

import pandas as pd
from pathlib import Path

# Load CSV
DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
csv = DATA_ROOT / "combined_candidates" / "weighted_score_candidates_050226.csv"
df = pd.read_csv(csv)

# Ensure blob_score is numeric (safe if it was read as text)
df["blob_score"] = pd.to_numeric(df["blob_score"], errors="coerce")

# Drop rows where blob_score is NaN
df = df.dropna(subset=["blob_score"])

# Pick best candidate per image (max blob_score)
best_idx = df.groupby("image")["blob_score"].idxmax()
best_df = df.loc[best_idx].sort_values(["image", "candidate_num"]).reset_index(drop=True)

# Save
out_csv = DATA_ROOT / "combined_candidates" / "best_candidates_050226.csv"
best_df.to_csv(out_csv, index=False)