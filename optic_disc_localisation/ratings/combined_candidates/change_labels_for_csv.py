import pandas as pd
from pathlib import Path
 
disc_localisation_path = Path(__file__).parents[1]
input_csv  = disc_localisation_path / "combined_candidates" / "best_candidates_050226.csv"
output_csv = disc_localisation_path / "combined_candidates" / "best_candidates_reclassified.csv"
 
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