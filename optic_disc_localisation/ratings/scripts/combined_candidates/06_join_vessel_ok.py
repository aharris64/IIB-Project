"""Step 6: merges the vessel_ok column (from the best-candidate manual ratings) into
all four combined-candidates CSVs, overwriting them in place — the one script that
bridges the best-candidate and combined-candidates rating trees."""

import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parents[2] / "data"

vessel_ok_csv = root / "best_candidate_ratings_trial_weights" / "disc_localisation_030226_manual.csv"

combined_candidates_dir = root / "combined_candidates"

TARGET_FILES = [
    combined_candidates_dir / "all_candidates_050226.csv",
    combined_candidates_dir / "weighted_score_candidates_050226.csv",
    combined_candidates_dir / "best_candidates_050226.csv",
    combined_candidates_dir / "final_candidates_090226.csv",
]

vessel_ok = pd.read_csv(vessel_ok_csv, usecols=["image", "vessel_ok"])

# Strip extension so "normal_0001_EDD.jpg" -> "normal_0001_EDD"
vessel_ok["image"] = vessel_ok["image"].str.replace(r"\.[^.]+$", "", regex=True)
vessel_ok = vessel_ok.drop_duplicates(subset="image")

for path in TARGET_FILES:
    df = pd.read_csv(path)

    # Drop existing vessel_ok if present (safe to re-run)
    if "vessel_ok" in df.columns:
        df = df.drop(columns=["vessel_ok"])

    df = df.merge(vessel_ok, on="image", how="left")

    # Place vessel_ok immediately after rating
    cols = df.columns.tolist()
    cols.remove("vessel_ok")
    cols.insert(cols.index("rating") + 1, "vessel_ok")
    df = df[cols]

    df.to_csv(path, index=False)

    matched = df["vessel_ok"].notna().sum()
    print(f"{path.name}: {len(df)} rows, {matched}/{len(df)} with vessel_ok")