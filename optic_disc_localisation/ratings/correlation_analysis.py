import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

name = "disc_localisation_030226.csv"

disc_localisation_path = Path(__file__).parents[1]
combined_csv = disc_localisation_path / "ratings" / "combined_results" / name

df = pd.read_csv(combined_csv)

cols = ["rating", "blob_contrast", "blob_brightness",
        "blob_response", "final_score"]

corr = df[cols].corr()

print(corr["rating"])

features = [
    "blob_contrast",
    "blob_brightness",
    "blob_response",
    "final_score"
]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(df[features])

# Correlation weights (from your results)
weights = corr.loc[features, "rating"].to_numpy()

print(weights)

# Weighted score
df["weighted_score"] = X_scaled @ weights

corr_new = df["weighted_score"].corr(df["rating"])
print("New correlation:", corr_new)

weights = np.array([ 12.2226, # contrast 
                   -6.6519, # brightness 
                    -23.0518, # response 
                    0.7543 # final 
                    ])

print(weights)

# Weighted score
df["weighted_score"] = X_scaled @ weights

corr_new = df["weighted_score"].corr(df["rating"])
print("New correlation:", corr_new)