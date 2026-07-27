"""Step 5: reclassifies papilledema rows whose image name suggests non-specific disc
oedema (EDD/IFD/RFM acronyms) into their own class, feeding
09_plot_best_candidate_properties.py."""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "combined_candidates"
input_csv  = DATA_DIR / "best_candidates_050226.csv"
output_csv = DATA_DIR / "best_candidates_reclassified.csv"
 
df = pd.read_csv(input_csv)
 
# Reclassify papilledema rows with EDD, IFD, or RFM suffix
reclassify_mask = (
    (df["class"] == "papilledema") &
    (df["image"].str.contains("EDD|IFD|RFM", regex=True))
)
 
df.loc[reclassify_mask, "class"] = "non_specific_disc_oedema"
 
# Summary
print("Class distribution after reclassification:")
print(df["class"].value_counts().to_string())
print(f"\nRows reclassified: {reclassify_mask.sum()}")
 
df.to_csv(output_csv, index=False)
print(f"\nSaved to {output_csv}")