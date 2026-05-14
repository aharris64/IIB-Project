from pathlib import Path
from PIL import Image

source = r"C:\Users\adam6\OneDrive\Documents\University\Engineering\Engineering IIB\IIB Project\Datasets\Processed Datasets\centred\disc_centred_r4.0_cl4"
destination = r"C:\Users\adam6\OneDrive\Documents\University\Engineering\Engineering IIB\IIB Project\Datasets\Processed Datasets"

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