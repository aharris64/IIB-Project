"""Exploratory only, not part of the reproducible pipeline: generates single-channel
H/S/V variants of the disc-centred images, to visually assess whether isolating an
HSV component might help optic disc localisation (ODL) pipeline development."""

import os
from pathlib import Path
from PIL import Image

DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "./Datasets")
source = os.path.join(DATASETS_ROOT, "Dataset", "basic_resize_224")
destination = os.path.join(DATASETS_ROOT, "Dataset", "hsv_test")

SOURCE_ROOT = Path(source)
DEST_ROOT = Path(destination)

CLASSES = ["normal", "papilledema", "pseudopapilledema"]
IMAGE_EXTS = {".jpg", ".png"}

CHANNELS = {
    "h_disc_centred_r4.0_cl4": lambda h, s, v: (h, h, h),
    "s_disc_centred_r4.0_cl4": lambda h, s, v: (s, s, s),
    "v_disc_centred_r4.0_cl4": lambda h, s, v: (v, v, v),
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
            img_hsv = img.convert("HSV")
            h, s, v = img_hsv.split()

            for ch, merge_fn in CHANNELS.items():
                out_img = Image.merge("RGB", merge_fn(h, s, v))
                out_img.save(DEST_ROOT / ch / cls / img_path.name)