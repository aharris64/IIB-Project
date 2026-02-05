import json
import pandas as pd
from pathlib import Path

disc_localisation_path = Path(__file__).parents[1]
disc_results_path = disc_localisation_path / "outputs" / "all_candidates_results.json"
csv_results_path = disc_localisation_path / "ratings" / "disc_candidates_050226.csv"


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