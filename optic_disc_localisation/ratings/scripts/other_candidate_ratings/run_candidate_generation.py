"""Batch driver for the "other candidates" rating set: for each raw dataset image,
finds every non-best blob candidate (find_other_candidates.detect_disc), saves
per-candidate overlay images, and writes a combined all_candidates_results.json."""

import os
from pathlib import Path
import json

from optic_disc_localisation.ratings.scripts.other_candidate_ratings.find_other_candidates import detect_disc

DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "./Datasets")
source = os.path.join(DATASETS_ROOT, "Processed Datasets", "raw")
destination = os.path.join(DATASETS_ROOT, "Processed Datasets", "disc_candidates050226")

SOURCE_ROOT = Path(source)
DEST_ROOT = Path(destination)

CLASSES = ["normal", "papilledema", "pseudopapilledema"]
IMAGE_EXTS = {".jpg", ".png"}

DEST_ROOT.mkdir(parents=True, exist_ok=True)

TARGET_SIZE=512

# parents[3] reaches optic_disc_localisation/, matching the real outputs/ folder that
# disc_localisaton_stats.py reads from.
disc_localisation_path = Path(__file__).resolve().parents[3]
out_file = disc_localisation_path / "outputs" / "all_candidates_results.json"
out_file.parent.mkdir(parents=True, exist_ok=True)

results = {}

for cls in CLASSES:
    src_cls_dir = SOURCE_ROOT / cls
    dst_cls_dir = DEST_ROOT / cls
    dst_cls_dir.mkdir(parents=True, exist_ok=True)

    for img_path in src_cls_dir.iterdir():
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        candidate_list = detect_disc(
            img_path,
            dst_cls_dir,
            target_size=TARGET_SIZE,
        )

        # Key by relative path to avoid collisions across classes
        key = str(img_path.relative_to(SOURCE_ROOT)).replace("\\", "/")

        # Convert to JSON-serializable structure
        candidates_json = []
        for vessel_blob_score, blob, vessel_centre in candidate_list:
            blob_centre, blob_radius, blob_score = blob
            # blob_score expected: (total_score, contrast, brightness, response)

            candidates_json.append({
                "vessel_blob_score": float(vessel_blob_score),
                "centre": [float(blob_centre[0]), float(blob_centre[1])],
                "radius": float(blob_radius),
                "blob_score": None if blob_score is None else float(blob_score[0]),
                "blob_contrast": None if blob_score is None else float(blob_score[1]),
                "blob_brightness": None if blob_score is None else float(blob_score[2]),
                "blob_response": None if blob_score is None else float(blob_score[3]),
                "vessel_centre": None if vessel_centre is None else [
                    float(vessel_centre[0]), float(vessel_centre[1])
                ],
            })

        # Save per-image record
        results[key] = {
            "class": cls,
            "num_candidates": len(candidates_json),
            "candidates": candidates_json,
        }

# Write JSON
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)