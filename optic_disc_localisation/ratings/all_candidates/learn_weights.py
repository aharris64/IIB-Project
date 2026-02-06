import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


disc_localisation_path = Path(__file__).parents[1]
combined_csv = disc_localisation_path / "all_candidates" / "all_candidates_050226.csv"

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