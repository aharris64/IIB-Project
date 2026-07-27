"""Step 3: computes the fitted weighted score (weights from 02_learn_weights.py) for
every candidate and writes it in as blob_score, producing final_candidates_090226.csv
— the one actively-used output of this whole ratings/ pipeline (everything else here
is historical working-out that fed into this file)."""

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
DATA_ROOT = Path(__file__).resolve().parents[3] / "data"
csv = DATA_ROOT / "history" / "combined_candidates" / "all_candidates_050226.csv"
df = pd.read_csv(csv)

# Compute weighted score
df = calculate_weighted_score(df)

# Replace blob_score with weighted_score
df["blob_score"] = df["weighted_score"]

df = df.drop(columns=["weighted_score"])

# Save updated CSV — the one active file at data/ root (not under history/)
out_csv = DATA_ROOT / "final_candidates_090226.csv"
df.to_csv(out_csv, index=False)