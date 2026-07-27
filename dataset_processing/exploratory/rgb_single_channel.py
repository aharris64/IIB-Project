"""Exploratory only, not part of the reproducible pipeline: generates single-channel
R/G/B variants of the disc-centred images, to visually assess whether isolating a
colour channel might help optic disc localisation (ODL) pipeline development."""

import os
from pathlib import Path
from PIL import Image

DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "./Datasets")
source = os.path.join(DATASETS_ROOT, "Processed Datasets", "basic_resize_224")
destination = os.path.join(DATASETS_ROOT, "Processed Datasets", "rgb_test")

SOURCE_ROOT = Path(source)
DEST_ROOT = Path(destination)

CLASSES = ["normal", "papilledema", "pseudopapilledema"]
IMAGE_EXTS = {".jpg", ".png"}

CHANNELS = {
    "r_disc_centred_r4.0_cl4": lambda r, g, b: (r, r, r),
    "g_disc_centred_r4.0_cl4": lambda r, g, b: (g, g, g),
    "b_disc_centred_r4.0_cl4": lambda r, g, b: (b, b, b),
}

# Create directory structure
for ch in CHANNELS:
    for cls in CLASSES:
        (DEST_ROOT / ch / cls).mkdir(parents=True, exist_ok=True)

# Process images
for cls in CLASSES:
    src_dir = SOURCE_ROOT / cls
    print("Processing ", cls)

    for img_path in src_dir.iterdir():
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            r, g, b = img.split()

            for ch, merge_fn in CHANNELS.items():
                out_img = Image.merge("RGB", merge_fn(r, g, b))
                out_img.save(DEST_ROOT / ch / cls / img_path.name)