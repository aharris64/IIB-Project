from pathlib import Path
import json
import os
from PIL import Image

from optic_disc_localisation.runners.run_disc_localisation import detect_disc
from optic_disc_localisation.image_processing.initial_processing import resize
from optic_disc_localisation.visualisations.save_visualisations import save_vessel_centre_and_blob_candidate, save_centre_overlay


source = r"C:\Users\adam6\OneDrive\Documents\University\Engineering\Engineering IIB\IIB Project\Datasets\Processed Datasets\raw"
destination = r"C:\Users\adam6\OneDrive\Documents\University\Engineering\Engineering IIB\IIB Project\Datasets\Processed Datasets\disc_localisation030226"

SOURCE_ROOT = Path(source)
DEST_ROOT = Path(destination)

CLASSES = ["normal", "papilledema", "pseudopapilledema"]
IMAGE_EXTS = {".jpg", ".png"}

DEST_ROOT.mkdir(parents=True, exist_ok=True)

TARGET_SIZE=512

disc_localisation_path = Path(__file__).parents[1]
no_blob_dir = disc_localisation_path / "outputs" / "final_output_images" / "no_blob_detected"
neg_score_dir = disc_localisation_path / "outputs" / "final_output_images" / "inconsistent_vessel_and_blob"

no_blob_dir.mkdir(parents=True, exist_ok=True)
neg_score_dir.mkdir(parents=True, exist_ok=True)

results = {}

for cls in CLASSES:
    src_cls_dir = SOURCE_ROOT / cls
    dst_cls_dir = DEST_ROOT / cls
    dst_cls_dir.mkdir(parents=True, exist_ok=True)

    for img_path in src_cls_dir.iterdir():
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        status, score, blob, vessel_centre = detect_disc(
            img_path,
            target_size=TARGET_SIZE,
            save_final=True,
            save_final_path=dst_cls_dir,
            save_intermediate=False,
            save_intermediate_path=False
        )

        # Load + resize once (for saving failures)
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img = resize(img, TARGET_SIZE)

        # Case 1: No blob
        if status == "no_blob":

            save_centre_overlay(img, vessel_centre, no_blob_dir, img_path.name)

            results[img_path.name] = {
                "status": "no_blob",
                "class": cls,
                "centre": None,
                "radius": None,
                "blob_score": None,
                "blob_contrast": None,
                "blob_brightness": None,
                "blob_response": None,
                "final_score": None,
                "vessel_centre": None if vessel_centre is None else [
                    float(vessel_centre[0]), float(vessel_centre[1])
                ],
            }

            continue

        # Case 2: Negative score
        if status == "ok" and score is not None and score < 0:

            save_vessel_centre_and_blob_candidate(img, blob, vessel_centre, neg_score_dir, img_path.name)

            blob_centre, blob_radius, blob_score = blob

            results[img_path.name] = {
                "status": "negative_score",
                "class": cls,
                "centre": [float(blob_centre[0]), float(blob_centre[1])],
                "radius": float(blob_radius),
                "blob_score": None if blob_score is None else float(blob_score[0]),
                "blob_contrast": None if blob_score is None else float(blob_score[1]),
                "blob_brightness": None if blob_score is None else float(blob_score[2]),
                "blob_response": None if blob_score is None else float(blob_score[3]),
                "final_score": float(score),
                "vessel_centre": None if vessel_centre is None else [
                    float(vessel_centre[0]), float(vessel_centre[1])
                ],
            }

            continue

        # ---------- Case 3: Normal success ----------
        if status == "ok":

            blob_centre, blob_radius, blob_score = blob

            results[img_path.name] = {
                "status": "ok",
                "class": cls,
                "centre": [float(blob_centre[0]), float(blob_centre[1])],
                "radius": float(blob_radius),
                "blob_score": None if blob_score is None else float(blob_score[0]),
                "blob_contrast": None if blob_score is None else float(blob_score[1]),
                "blob_brightness": None if blob_score is None else float(blob_score[2]),
                "blob_response": None if blob_score is None else float(blob_score[3]),
                "final_score": float(score),
                "vessel_centre": None if vessel_centre is None else [
                    float(vessel_centre[0]), float(vessel_centre[1])
                ],
            }

# Write results JSON next to this script
out_file = disc_localisation_path / "outputs" / "disc_localisation_results.json"
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)