from pathlib import Path
from PIL import Image

source = r"C:\Users\adam6\OneDrive\Documents\University\Engineering\Engineering IIB\IIB Project\Datasets\Processed Datasets\disc_centred_r4.0_cl4"
destination = r"C:\Users\adam6\OneDrive\Documents\University\Engineering\Engineering IIB\IIB Project\Datasets\Processed Datasets"

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