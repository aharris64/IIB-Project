import pandas as pd
import ast
from pathlib import Path

disc_localisation_path = Path(__file__).parents[1]
generated_csv = disc_localisation_path / "candidate_ratings" / "disc_candidates_050226.csv"
manual_csv    = disc_localisation_path / "candidate_ratings" / "disc_candidates_050226_manual.csv"
outpath       = disc_localisation_path / "candidate_ratings" / "combined_candidates_050226.csv"

# --- load ---
df1 = pd.read_csv(generated_csv)   # filename,class,num_candidates,candidates
df2 = pd.read_csv(manual_csv)      # image,candidate_num,rating,index

# --- keep only rows with candidates ---
df1 = df1[df1["num_candidates"].fillna(0).astype(int) > 0].copy()

# --- build join key (strip folder + extension) ---
df1["image_key"] = (
    df1["filename"].astype(str)
      .str.replace(r"^.*/", "", regex=True)
      .str.replace(r"\.jpg$", "", regex=True)
)

df1["image_clean"] = (
    df1["filename"].astype(str)
      .str.replace(r"^.*/", "", regex=True)
      .str.replace(r"\.jpg$", "", regex=True)
)

# --- parse candidates list-of-dicts and expand to one row per candidate ---
df1["candidates"] = df1["candidates"].apply(
    lambda x: [] if pd.isna(x) or x == "[]" else ast.literal_eval(x)
)

df_long = df1.explode("candidates", ignore_index=True)

# drop any accidental empty rows (shouldn't happen if num_candidates > 0, but safe)
df_long = df_long[df_long["candidates"].notna()].copy()

# candidate_num = 1..N within each image, matching manual file convention
df_long["candidate_num"] = df_long.groupby("image_key").cumcount() + 1

# normalize dict columns
cand_cols = pd.json_normalize(df_long["candidates"])
df_long = pd.concat([df_long.drop(columns=["candidates"]), cand_cols], axis=1)

# --- clean manual keys (so normal_0009_EDD and normal_0009_EDD.jpg both work) ---
df2 = df2.copy()
df2["image_key"] = (
    df2["image"].astype(str)
      .str.replace(r"^.*/", "", regex=True)
      .str.replace(r"\.jpg$", "", regex=True)
)

# force numeric candidate_num if possible (handles weird entries safely)
df2["candidate_num"] = pd.to_numeric(df2["candidate_num"], errors="coerce").astype("Int64")
df2 = df2[df2["candidate_num"].notna()].copy()
df2["candidate_num"] = df2["candidate_num"].astype(int)

# --- merge ratings onto expanded candidates ---
out = df_long.merge(
    df2[["image_key", "candidate_num", "rating", "index"]],
    on=["image_key", "candidate_num"],
    how="inner",
)

# --- rename / compute output fields ---
out = out.rename(columns={
    "image_clean": "image",
    "class": "class_x",
    "index": "class_index",
    "vessel_blob_score": "final_score",
})

# # optional fields (edit to your real definitions)
# out["vessel_ok"] = out["final_score"] > 0
# out["status"] = out["rating"].apply(lambda r: "ok" if r >= 4 else "review")
# out["class_y"] = out["class_x"]

# --- final column order ---
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

# keep only columns that exist (prevents crash if some fields missing in candidates dict)
cols = [c for c in cols if c in out.columns]
out = out[cols]

out.to_csv(outpath, index=False)
print(f"Wrote {outpath} with {len(out)} rows")