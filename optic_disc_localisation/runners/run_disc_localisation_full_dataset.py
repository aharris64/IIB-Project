from pathlib import Path
import json
import os
from PIL import Image

from optic_disc_localisation.runners.run_disc_localisation import detect_disc
from optic_disc_localisation.image_processing.initial_processing import resize
from optic_disc_localisation.visualisations.save_visualisations import save_vessel_centre_and_blob_candidate, save_centre_overlay


source = r"C:\Users\adam6\OneDrive\Documents\University\Engineering\Engineering IIB\IIB Project\Datasets\Processed Datasets\raw"
destination = r"C:\Users\adam6\OneDrive\Documents\University\Engineering\Engineering IIB\IIB Project\Datasets\Processed Datasets\disc_localisation_100226_BIG"

SOURCE_ROOT = Path(source)
DEST_ROOT = Path(destination)

CLASSES = ["normal", "papilledema", "pseudopapilledema"]
IMAGE_EXTS = {".jpg", ".png"}

DEST_ROOT.mkdir(parents=True, exist_ok=True)

TARGET_SIZE=1280

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

        blob_centre, blob_radius, vessel_centre, scores = detect_disc(
            img_path,
            target_size=TARGET_SIZE,
            save_final=True,
            save_final_path=dst_cls_dir,
            save_intermediate=False,
            save_intermediate_path=False
        )

        total_score, contrast, response, vessel_sign = scores

        # Load + resize once (for saving failures)
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img = resize(img, TARGET_SIZE)

        if blob_centre is None:

            save_centre_overlay(img, vessel_centre, no_blob_dir, img_path.name)

            results[img_path.name] = {
                "class": cls,
                "centre": None,
                "radius": None,
                "vessel_centre": [float(vessel_centre[0]),float(vessel_centre[1])],
                "score": None,
                "contrast": None,
                "response": None,
                "vessel_sign": None,
            }

            continue

        # Case 2: Vessel and Blob don't match
        if blob_centre is not None and vessel_sign == 0:

            save_vessel_centre_and_blob_candidate(img, (blob_centre, blob_radius, None), vessel_centre, neg_score_dir, img_path.name)

            results[img_path.name] = {
                "class": cls,
                "centre": [float(blob_centre[0]), float(blob_centre[1])],
                "radius": float(blob_radius),
                "vessel_centre": [float(vessel_centre[0]),float(vessel_centre[1])],
                "score": float(total_score),
                "contrast": float(contrast),
                "response": float(response),
                "vessel_sign": bool(vessel_sign),
            }

            continue

        # Case 3: Normal success
        results[img_path.name] = {
            "class": cls,
            "centre": [float(blob_centre[0]), float(blob_centre[1])],
            "radius": float(blob_radius),
            "vessel_centre": [float(vessel_centre[0]),float(vessel_centre[1])],
            "score": float(total_score),
            "contrast": float(contrast),
            "response": float(response),
            "vessel_sign": bool(vessel_sign),
        }

# Write results JSON next to this script
out_file = disc_localisation_path / "outputs" / "disc_localisation_results.json"
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)