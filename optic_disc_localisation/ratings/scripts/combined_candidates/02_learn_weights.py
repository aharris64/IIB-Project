"""Step 2: fits a linear regression of manual rating on candidate features to derive
the weights hardcoded into combined_method/candidate_evaluation.py's score_candidate
(w_bias, w_contrast, w_response, w_vessel_sign) and 03_apply_weighted_score.py's
calculate_weighted_score. One-time calibration tool — its printed output is manually
copy-pasted into those two places, not consumed automatically by anything downstream."""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
combined_csv = DATA_ROOT / "combined_candidates" / "all_candidates_050226.csv"

df = pd.read_csv(combined_csv)

df["final_sign"] = (df["final_score"] > 0).astype(int) # Just use sign

FEATURES = [
    "blob_contrast",
    "blob_response",
    "final_sign",
]

# Drop rows with missing values
df = df.dropna(subset=FEATURES + ["rating"]).copy()

# Ensure numeric
for c in FEATURES + ["rating"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=FEATURES + ["rating"]).copy()

X = df[FEATURES]
y = 5 - df["rating"].astype(float)   # higher rating = better

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=df["rating"]
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("reg", LinearRegression())
])

model.fit(X_train, y_train)

pred = model.predict(X_test)

corr = np.corrcoef(pred, y_test)[0, 1]
mae = mean_absolute_error(y_test, pred)

print("Test correlation:", corr)
print("Test MAE:", mae)

# Extract weights
scaler = model.named_steps["scaler"]
reg = model.named_steps["reg"]

# Convert back from standardized space
w = reg.coef_ / scaler.scale_
bias = reg.intercept_ - np.dot(w, scaler.mean_)

print("\nLearned weights (higher score = better):")

for f, val in zip(FEATURES, w):
    print(f"{f}: {val:.6f}")

print("bias:", bias)