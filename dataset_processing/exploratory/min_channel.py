"""Exploratory only, not part of the reproducible pipeline: generates a min(R, G, B)
single-channel variant of the basic-resized images, to visually assess whether this
representation might help optic disc localisation (ODL) pipeline development."""

import os
from pathlib import Path
from PIL import Image, ImageChops

DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "./Datasets")
source = os.path.join(DATASETS_ROOT, "Dataset", "basic_resize_224")
destination = os.path.join(DATASETS_ROOT, "Dataset", "min_test")

SOURCE_ROOT = Path(source)
DEST_ROOT = Path(destination)

CLASSES = ["normal", "papilledema", "pseudopapilledema"]
IMAGE_EXTS = {".jpg", ".png"}

# Process images
for cls in CLASSES:
    src_dir = SOURCE_ROOT / cls

    for img_path in src_dir.iterdir():
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            r, g, b = img.split()

            min_img = ImageChops.darker(ImageChops.darker(r, g), b)
            out_img = Image.merge("RGB", (min_img, min_img, min_img))
            out_img.save(DEST_ROOT / cls / img_path.name)