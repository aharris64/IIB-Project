import pandas as pd
from pathlib import Path

# Load CSV
disc_localisation_path = Path(__file__).parents[1]
csv = disc_localisation_path / "all_candidates" / "weighted_score_candidates_050226.csv"
df = pd.read_csv(csv)

# Ensure blob_score is numeric (safe if it was read as text)
df["blob_score"] = pd.to_numeric(df["blob_score"], errors="coerce")

# Drop rows where blob_score is NaN
df = df.dropna(subset=["blob_score"])

# Pick best candidate per image (max blob_score)
best_idx = df.groupby("image")["blob_score"].idxmax()
best_df = df.loc[best_idx].sort_values(["image", "candidate_num"]).reset_index(drop=True)

# Save
out_csv = disc_localisation_path / "all_candidates" / "best_candidates_050226.csv"
best_df.to_csv(out_csv, index=False)