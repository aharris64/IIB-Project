"""Step 7 (terminal/standalone report): reports how often the highest-weighted-score
candidate matches the best-rated candidate per image, overall/by-class/by-rating-gap,
plus the worst failure cases."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


def calculate_weighted_score(df):
    w_contrast = 3.83144
    w_response = -8.45457
    w_final    = 1.211922
    bias       = 0.5929768734602114

    df["final_sign"]    = (df["final_score"] > 0).astype(int)
    df["weighted_score"] = (
        w_contrast * df["blob_contrast"]
        + w_response * df["blob_response"]
        + w_final    * df["final_sign"]
        + bias
    )
    return df


def analyse_candidate_selection(df):
    """
    Per image:
      - selected     : candidate with highest weighted_score
      - best_possible: candidate with lowest rating (best localisation)
      - missed       : selected rating > best_possible rating
      - no_candidate : all candidates rated 3 or 4
    """
    records = []

    for image, group in df.groupby("image"):
        # Drop rows where weighted_score is NaN before selecting
        valid = group.dropna(subset=["weighted_score"])
        
        if valid.empty:
            # No valid candidates at all — skip or mark as failed
            records.append({
                "image":                image,
                "class":               group["class"].iloc[0],
                "n_candidates":        len(group),
                "selected_rating":     4,   # treat as complete failure
                "best_possible_rating": 4,
                "missed":              0,
                "no_good_candidate":   1,
                "rating_gap":          0,
            })
            continue

        selected    = valid.loc[valid["weighted_score"].idxmax()]
        best_rating = valid["rating"].min()
        missed      = int(selected["rating"] > best_rating)
        no_good_cand = int(best_rating >= 3)

        records.append({
            "image":                image,
            "class":               selected["class"],
            "n_candidates":        len(group),
            "selected_rating":     selected["rating"],
            "best_possible_rating": best_rating,
            "missed":              missed,
            "no_good_candidate":   no_good_cand,
            "rating_gap":          selected["rating"] - best_rating,
        })

    return pd.DataFrame(records)


def print_candidate_analysis(df):
    df = calculate_weighted_score(df)
    summary = analyse_candidate_selection(df)

    classes = sorted(summary["class"].unique())
    total   = len(summary)

    def section(title):
        print(f"\n{'─'*60}")
        print(f"  {title}")
        print(f"{'─'*60}")

    # ── Overall ───────────────────────────────────────────────────────
    section("OVERALL")
    print(f"  Total images:              {total}")
    print(f"  Best candidate selected:   {(summary['missed']==0).sum()}  ({(summary['missed']==0).mean()*100:.1f}%)")
    print(f"  Better candidate missed:   {(summary['missed']==1).sum()}  ({(summary['missed']==1).mean()*100:.1f}%)")
    print(f"  Good localisation (1,2):   {summary['selected_rating'].isin([1,2]).sum()}  ({summary['selected_rating'].isin([1,2]).mean()*100:.1f}%)")
    print(f"  Poor localisation (3,4):   {summary['selected_rating'].isin([3,4]).sum()}  ({summary['selected_rating'].isin([3,4]).mean()*100:.1f}%)")

    # ── Rating gap ────────────────────────────────────────────────────
    section("RATING GAP (selected − best available)")
    for gap in sorted(summary["rating_gap"].unique()):
        n = (summary["rating_gap"] == gap).sum()
        print(f"  Gap {gap}:  {n:5d}  ({n/total*100:.1f}%)")

    # ── Why did poor localisations fail? ─────────────────────────────
    section("CAUSE OF POOR LOCALISATION (selected rating 3 or 4)")
    poor = summary[summary["selected_rating"] >= 3]
    print(f"  Total poor:                {len(poor)}  ({len(poor)/total*100:.1f}% of all images)")
    no_cand  = poor["no_good_candidate"].sum()
    missed_g = len(poor) - no_cand
    print(f"  No good candidate existed: {no_cand}  ({no_cand/len(poor)*100:.1f}% of poor)")
    print(f"  Good candidate was missed: {missed_g}  ({missed_g/len(poor)*100:.1f}% of poor)")

    # ── Per class breakdown ───────────────────────────────────────────
    section("PER CLASS BREAKDOWN")
    for cls in classes:
        sub  = summary[summary["class"] == cls]
        poor = sub[sub["selected_rating"] >= 3]
        print(f"\n  {cls.upper()}  (n={len(sub)})")
        print(f"    Good localisation:         {sub['selected_rating'].isin([1,2]).sum():4d}  ({sub['selected_rating'].isin([1,2]).mean()*100:.1f}%)")
        print(f"    Poor localisation:         {sub['selected_rating'].isin([3,4]).sum():4d}  ({sub['selected_rating'].isin([3,4]).mean()*100:.1f}%)")
        print(f"    Best candidate selected:   {(sub['missed']==0).sum():4d}  ({(sub['missed']==0).mean()*100:.1f}%)")
        if len(poor) > 0:
            no_cand  = poor["no_good_candidate"].sum()
            missed_g = len(poor) - no_cand
            print(f"    Poor — no good candidate:  {no_cand:4d}  ({no_cand/len(poor)*100:.1f}% of poor)")
            print(f"    Poor — good was missed:    {missed_g:4d}  ({missed_g/len(poor)*100:.1f}% of poor)")

    # ── Worst failures ────────────────────────────────────────────────
    section("WORST FAILURES (gap = 3, good candidate existed but missed)")
    worst = summary[(summary["rating_gap"] == 3) & (summary["missed"] == 1)]
    if len(worst) > 0:
        print(worst[["image", "class", "selected_rating",
                      "best_possible_rating", "rating_gap"]].to_string(index=False))
    else:
        print("  None.")

    return summary


# ── Run ───────────────────────────────────────────────────────────────
weighted_candidates_path = Path(__file__).resolve().parents[2] / "data"
csv = weighted_candidates_path / "combined_candidates" / "weighted_score_candidates_050226.csv"

df = pd.read_csv(csv)
summary = print_candidate_analysis(df)