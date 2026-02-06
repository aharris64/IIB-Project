import pandas as pd
from pathlib import Path
import os

name = "disc_localisation_030226.csv"

disc_localisation_path = Path(__file__).parents[1]
manual_csv = disc_localisation_path / "ratings" / "manual_ratings" / name
generated_csv = disc_localisation_path / "ratings" / "csv_results" / name

save_path = disc_localisation_path / "ratings" / "combined_results" / name

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