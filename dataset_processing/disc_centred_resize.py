
from pathlib import Path
import json
import pandas as pd
from PIL import Image, ImageOps

CLASSES = ["normal", "papilledema", "pseudopapilledema"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

TARGET_SHORT = 512
OUT_SIZE = 224
PAD_COLOR = (0, 0, 0)
RADIUS_SCALE_FACTOR = 4.0
ACCEPT_THRESHOLD = 1.75


name = f"disc_centred_r{RADIUS_SCALE_FACTOR}_th{ACCEPT_THRESHOLD}"

source = Path(r"C:\Users\adam6\OneDrive\Documents\University\Engineering\Engineering IIB\IIB Project\Datasets\Processed Datasets\raw")
dest   = Path(r"C:\Users\adam6\OneDrive\Documents\University\Engineering\Engineering IIB\IIB Project\Datasets\Processed Datasets\disc_centred_r4")
dest.mkdir(parents=True, exist_ok=True)

json_file = Path(r"C:\Users\adam6\OneDrive\Documents\University\Engineering\Engineering IIB\IIB Project\Datasets\Processed Datasets\disc_localisation_100226_512\disc_localisation_results.json")

def resize_scale_factor(w: int, h: int, target_short: int) -> float:
    # matches resize(): if both sides smaller, no resize
    if w < target_short and h < target_short:
        return 1.0
    return target_short / min(w, h)

def centred_square_box(cx: float, cy: float, side: float):
    half = side / 2.0
    return (cx - half, cy - half, cx + half, cy + half)

def crop_with_padding(img: Image.Image, box):
    left, top, right, bottom = box
    w, h = img.size

    pad_left   = max(0, int(round(-left)))
    pad_top    = max(0, int(round(-top)))
    pad_right  = max(0, int(round(right - w)))
    pad_bottom = max(0, int(round(bottom - h)))

    if any(p > 0 for p in (pad_left, pad_top, pad_right, pad_bottom)):
        img = ImageOps.expand(img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=PAD_COLOR)
        left += pad_left; right += pad_left
        top  += pad_top;  bottom += pad_top

    crop_box = (int(round(left)), int(round(top)), int(round(right)), int(round(bottom)))
    return img.crop(crop_box)

with open(json_file, "r") as f:
    data = json.load(f)

df = pd.DataFrame.from_dict(data, orient="index")  # index should be filenames

for cls in CLASSES:
    (dest / cls).mkdir(parents=True, exist_ok=True)
    src_dir = source / cls
    dst_dir = dest / cls

    for img_path in src_dir.iterdir():
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        name = img_path.name
        if name not in df.index:
            print(f"[WARN] {cls}/{name} not in JSON; skipping.")
            continue

        if df.loc[name, "centre"] == None:
            print(f"[WARN] No blob detected")
            continue

        cx_512, cy_512 = df.loc[name, "centre"]
        r_512 = float(df.loc[name, "radius"])

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            w0, h0 = img.size

            s = resize_scale_factor(w0, h0, TARGET_SHORT)  # original -> 512 space
            # invert mapping: 512 space -> original
            cx0 = float(cx_512) / s
            cy0 = float(cy_512) / s
            r0  = float(r_512)  / s

            side0 = RADIUS_SCALE_FACTOR * r0  # 3 * diameter

            crop = crop_with_padding(img, centred_square_box(cx0, cy0, side0))
            crop = crop.resize((OUT_SIZE, OUT_SIZE), Image.LANCZOS)

            crop.save(dst_dir / name)
