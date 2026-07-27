"""Merges the generated (auto-detected) candidates with their manual ratings, one row
per candidate. NOTE: paths below were previously broken (phantom "candidate_ratings"
subfolder, and a generated-file name mismatch) — fixed to the real suffix-based
filenames (_generated/_manual/_combined) living directly in this script's own data/
folder, mirroring combine_csv_best_trial_weights.py's equivalent fix for the other
rating tree."""

import pandas as pd
import ast
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "other_candidate_ratings"
generated_csv = DATA_DIR / "disc_candidates_050226_generated.csv"
manual_csv    = DATA_DIR / "disc_candidates_050226_manual.csv"
outpath       = DATA_DIR / "disc_candidates_050226_combined.csv"

# --- load ---
df_generated = pd.read_csv(generated_csv)   # filename,class,num_candidates,candidates
df_manual = pd.read_csv(manual_csv)      # image,candidate_num,rating,index

# --- keep only rows with candidates ---
df_generated = df_generated[df_generated["num_candidates"].fillna(0).astype(int) > 0].copy()

# build join key (strip folder + extension)
df_generated["image_key"] = (
    df_generated["filename"].astype(str)
      .str.replace(r"^.*/", "", regex=True)
      .str.replace(r"\.jpg$", "", regex=True)
)

# clean generated keys (so normal_0009_EDD and normal_0009_EDD.jpg both work)
df_generated["image_clean"] = (
    df_generated["filename"].astype(str)
      .str.replace(r"^.*/", "", regex=True)
      .str.replace(r"\.jpg$", "", regex=True)
)

# parse candidates list-of-dicts and expand to one row per candidate
df_generated["candidates"] = df_generated["candidates"].apply(
    lambda x: [] if pd.isna(x) or x == "[]" else ast.literal_eval(x)
)

df_long = df_generated.explode("candidates", ignore_index=True)

# drop any accidental empty rows (shouldn't happen if num_candidates > 0, but safe)
df_long = df_long[df_long["candidates"].notna()].copy()

# candidate_num = 1..N within each image, matching manual file convention
df_long["candidate_num"] = df_long.groupby("image_key").cumcount() + 1

# normalize dict columns
cand_cols = pd.json_normalize(df_long["candidates"])
df_long = pd.concat([df_long.drop(columns=["candidates"]), cand_cols], axis=1)

# clean manual keys (so normal_0009_EDD and normal_0009_EDD.jpg both work)
df_manual = df_manual.copy()
df_manual["image_key"] = (
    df_manual["image"].astype(str)
      .str.replace(r"^.*/", "", regex=True)
      .str.replace(r"\.jpg$", "", regex=True)
)

# force numeric candidate_num if possible (handles weird entries safely)
df_manual["candidate_num"] = pd.to_numeric(df_manual["candidate_num"], errors="coerce").astype("Int64")
df_manual = df_manual[df_manual["candidate_num"].notna()].copy()
df_manual["candidate_num"] = df_manual["candidate_num"].astype(int)

# merge ratings onto expanded candidates
out = df_long.merge(
    df_manual[["image_key", "candidate_num", "rating", "index"]],
    on=["image_key", "candidate_num"],
    how="inner",
)

# rename output fields
out = out.rename(columns={
    "image_clean": "image",
    "class": "class_x",
    "index": "class_index",
    "vessel_blob_score": "final_score",
})

# final column order
cols = [
    "image",
    "class_x",
    "candidate_num",
    "rating",
    "class_index",
    "centre",
    "radius",
    "blob_score",
    "blob_contrast",
    "blob_brightness",
    "blob_response",
    "final_score",
    "vessel_centre",
]

# keep only columns that exist
cols = [c for c in cols if c in out.columns]
out = out[cols]

out.to_csv(outpath, index=False)
print(f"Wrote {outpath} with {len(out)} rows")