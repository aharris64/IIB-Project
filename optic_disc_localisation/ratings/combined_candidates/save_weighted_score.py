import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def calculate_weighted_score(df):
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

# Load CSV
disc_localisation_path = Path(__file__).parents[1]
csv = disc_localisation_path / "all_candidates" / "all_candidates_050226.csv"
df = pd.read_csv(csv)

# Compute weighted score
df = calculate_weighted_score(df)

# Replace blob_score with weighted_score
df["blob_score"] = df["weighted_score"]

df = df.drop(columns=["weighted_score"])

# Save updated CSV
out_csv = disc_localisation_path / "all_candidates" / "final_candidates_090226.csv"
df.to_csv(out_csv, index=False)