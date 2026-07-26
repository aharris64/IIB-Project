"""Converts the all_candidates_results.json produced by
ratings/scripts/other_candidate_ratings/run_candidate_generation.py into a flat CSV.
NOTE: paths below were previously broken — the JSON read path was off-by-one (missed
the optic_disc_localisation/ level entirely) and the CSV write path was
self-referentially nested ("ratings/ratings/..."). The read fix is high-confidence
(exact existing file); the write destination is a new file (no existing file
obviously matched it), so double-check this one specifically."""

import json
import pandas as pd
from pathlib import Path

disc_localisation_path = Path(__file__).resolve().parents[3]  # optic_disc_localisation/
disc_results_path = disc_localisation_path / "outputs" / "all_candidates_results.json"
csv_results_path = Path(__file__).resolve().parents[2] / "data" / "best_candidate_ratings_trial_weights" / "disc_candidates_050226_from_json.csv"


# Load JSON file
with open(disc_results_path, "r") as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame.from_dict(data, orient="index")

# Reset index to make filename a column
df.reset_index(inplace=True)
df.rename(columns={"index": "filename"}, inplace=True)

# Save to CSV
df.to_csv(csv_results_path, index=False)

print("Converted to output.csv")