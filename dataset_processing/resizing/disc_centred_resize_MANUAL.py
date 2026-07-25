
import os
from pathlib import Path
import json
import pandas as pd
from PIL import Image, ImageOps


# ----------------------------------------------------------------------------
# ------- WILL ONLY WORK ON MANUALLY RATED DISC (I.E. 512 TARGET_SIZE) -------
# ----------------------------------------------------------------------------

CLASSES = ["normal", "papilledema", "pseudopapilledema"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

TARGET_SIZE = 512
OUT_SIZE = 224
PAD_COLOR = (0, 0, 0)
RADIUS_SCALE_FACTOR = 4.0
REJECT_CLASS = [3, 4]


name = f"disc_centred_r{RADIUS_SCALE_FACTOR}_cl{''.join(str(c) for c in REJECT_CLASS)}"

DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "./Datasets")
source = Path(DATASETS_ROOT) / "Processed Datasets" / "raw"
dest_root   = Path(DATASETS_ROOT) / "Processed Datasets"
dest = dest_root / name
dest.mkdir(parents=True, exist_ok=True)

json_file = Path(DATASETS_ROOT) / "Processed Datasets" / "disc_localisation" / "disc_localisation_100226_512" / "disc_localisation_results.json"

# Load CSV
path = Path(__file__).parents[1]
csv = path / "optic_disc_localisation" / "ratings" / "combined_candidates" / "best_candidates_050226.csv"
df_ratings = pd.read_csv(csv)
df_ratings = df_ratings.set_index("image")

def resize_scale_factor(w: int, h: int, target_size: int) -> float:
    # matches resize(): if both sides smaller, no resize
    if w < target_size and h < target_size:
        return 1.0
    return target_size / min(w, h)

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

        name_clean = Path(name).stem

        if df_ratings.loc[name_clean, "rating"] in REJECT_CLASS:
            print(f"[WARN] Disc Localisation Classed as Poor")
            continue

        cx_512, cy_512 = df.loc[name, "centre"]
        r_512 = float(df.loc[name, "radius"])

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            w0, h0 = img.size

            s = resize_scale_factor(w0, h0, TARGET_SIZE)  # original -> 512 space
            # invert mapping: 512 space -> original
            cx0 = float(cx_512) / s
            cy0 = float(cy_512) / s
            r0  = float(r_512)  / s

            side0 = RADIUS_SCALE_FACTOR * r0  # 3 * diameter

            crop = crop_with_padding(img, centred_square_box(cx0, cy0, side0))
            crop = crop.resize((OUT_SIZE, OUT_SIZE), Image.LANCZOS)

            crop.save(dst_dir / name)
