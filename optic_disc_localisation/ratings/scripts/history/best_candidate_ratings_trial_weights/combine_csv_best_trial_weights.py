"""Merges the manual ratings with the generated candidate results, producing the
combined CSV"""

import pandas as pd
from pathlib import Path

stem = "disc_localisation_030226"

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "history" / "best_candidate_ratings_trial_weights"
manual_csv = DATA_DIR / f"{stem}_manual.csv"
generated_csv = DATA_DIR / f"{stem}_generated.csv"

save_path = DATA_DIR / f"{stem}_combined.csv"

# Load the CSV files
df1 = pd.read_csv(manual_csv)   # has: image, class, rating, ...
df2 = pd.read_csv(generated_csv)   # has: filename, status, centre, ...

# Merge on image name
combined = pd.merge(
    df1,
    df2,
    left_on="image",
    right_on="filename",
    how="inner"   # keeps only matching rows
)

combined = combined.drop(columns=["filename"])

# Save result
combined.to_csv(save_path, index=False)

print(combined.head())