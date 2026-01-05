import pandas as pd
from pathlib import Path

file = Path(__file__).resolve().parent / "ratings.csv"
df = pd.read_csv(file)

N = len(df)

class_counts = df["class"].value_counts().sort_index()
rating_counts = df["rating"].value_counts().sort_index()

class_pct = 100 * class_counts / N
rating_pct = 100 * rating_counts / N

table_counts = pd.crosstab(df["class"], df["rating"])
table_pct = 100 * table_counts.div(table_counts.sum(axis=1), axis=0)

print("\nClass × Rating (counts)")
print(table_counts)

print("\nClass × Rating (row %)")
print(table_pct.round(2))