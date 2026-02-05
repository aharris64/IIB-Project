import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error


# ============================
# EDIT THIS
# ============================


name = "disc_localisation_030226.csv"

disc_localisation_path = Path(__file__).parents[1]
combined_csv = disc_localisation_path / "ratings" / "combined_results" / name

CSV_PATH = combined_csv   # <-- put your filename here


# Features to use
FEATURES = [
    "blob_contrast",
    "blob_brightness",
    "blob_response",
    "final_score"
]

# FEATURES = [
#     "blob_contrast",
#     "blob_brightness",
#     "blob_response",
# ]


# ============================
# Load data
# ============================
df = pd.read_csv(CSV_PATH)

# Drop rows with missing values
df = df.dropna(subset=FEATURES + ["rating"]).copy()

# Ensure numeric
for c in FEATURES + ["rating"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=FEATURES + ["rating"]).copy()


# ============================
# Prepare X / y
# ============================
X = df[FEATURES]
y = 5 - df["rating"].astype(float)   # higher = better


# ============================
# Train / test split
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=df["rating"]
)


# ============================
# Model: Scaler + Ridge
# ============================
model = Pipeline([
    ("scaler", StandardScaler()),
    ("reg", Ridge(alpha=1.0))   # alpha = regularization strength
])

model.fit(X_train, y_train)


# ============================
# Evaluate
# ============================
pred = model.predict(X_test)

corr = np.corrcoef(pred, y_test)[0, 1]
mae = mean_absolute_error(y_test, pred)

print("Test correlation:", corr)
print("Test MAE:", mae)


# ============================
# Extract weights (original units)
# ============================
scaler = model.named_steps["scaler"]
reg = model.named_steps["reg"]

# Convert back from standardized space
w = reg.coef_ / scaler.scale_
bias = reg.intercept_ - np.dot(w, scaler.mean_)

print("\nLearned weights (higher score = better):")

for f, val in zip(FEATURES, w):
    print(f"{f}: {val:.6f}")

print("bias:", bias)


# ============================
# Example: scoring candidates
# ============================
def score_candidates(candidates_df):
    """
    candidates_df must have the same feature columns.
    Returns predicted rating (higher = better).
    """
    Xc = candidates_df[FEATURES].to_numpy(dtype=float)
    return Xc @ w + bias


# Example usage (commented out):
#
# candidates = pd.read_csv("candidates_one_image.csv")
# scores = score_candidates(candidates)
# best_idx = scores.argmin()
# print("Best candidate index:", best_idx)
