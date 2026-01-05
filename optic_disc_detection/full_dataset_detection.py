from pathlib import Path
from PIL import Image
import json
import os

from disc_detection import detect_disc
from initial_processing import resize
from save_results import save_image, save_candidate_overlay

source = r"C:\Users\adam6\OneDrive\Documents\University\Engineering\Engineering IIB\IIB Project\Datasets\Dataset\raw"
destination = r"C:\Users\adam6\OneDrive\Documents\University\Engineering\Engineering IIB\IIB Project\Datasets\Dataset\disc_detection"


SOURCE_ROOT = Path(source)
DEST_ROOT = Path(destination)

CLASSES = ["normal", "papilledema", "pseudopapilledema"]
TARGET_SIZE = 512
IMAGE_EXTS = {".jpg", ".png"}

DEST_ROOT.mkdir(parents=True, exist_ok=True)

results = {}

for cls in CLASSES:
    src_cls_dir = SOURCE_ROOT / cls
    dst_cls_dir = DEST_ROOT / cls
    dst_cls_dir.mkdir(parents=True, exist_ok=True)

    for img_path in src_cls_dir.iterdir():
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            resize_img = resize(img, TARGET_SIZE)

        best = detect_disc(resize_img)

        if best is not None:
            centre, radius, _ = best
            results[img_path.name] = {
                "centre": [float(centre[0]), float(centre[1])],
                "radius": float(radius),
                "class": cls
            }
            save_candidate_overlay(resize_img, [best], dst_cls_dir, img_path.name)
        else:
            current_path = Path(__file__).resolve().parent
            no_disc = os.path.join(current_path, "no_disc_detected")
            save_image(resize_img, no_disc, img_path.name)
            results[img_path.name] = None

save_disc_dict = Path(__file__).resolve().parent
out_file = save_disc_dict / "disc_detections.json"

with open(out_file, "w") as f:
    json.dump(results, f, indent=2)

# Code to open json file
# with open(out_file, "r") as f:
#     results = json.load(f)
# entry = results.get("example.jpg")
# if entry is not None:
#     centre = tuple(entry["centre"])
#     radius = entry["radius"]