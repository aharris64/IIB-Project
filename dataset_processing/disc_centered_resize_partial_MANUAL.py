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

# Quarter crop overlap: 0.0 = exact quadrants, 0.1 = 10% overlap between quarters
OVERLAP_FRAC = 0.0

name = f"disc_centred_r{RADIUS_SCALE_FACTOR}_cl{''.join(str(c) for c in REJECT_CLASS)}_quarters_overlap{OVERLAP_FRAC}"

source    = Path(r"C:\Users\adam6\OneDrive\Documents\University\Engineering\Engineering IIB\IIB Project\Datasets\Processed Datasets\raw")
dest_root = Path(r"C:\Users\adam6\OneDrive\Documents\University\Engineering\Engineering IIB\IIB Project\Datasets\Processed Datasets")
dest      = dest_root / name
dest.mkdir(parents=True, exist_ok=True)

json_file = Path(r"C:\Users\adam6\OneDrive\Documents\University\Engineering\Engineering IIB\IIB Project\Datasets\Processed Datasets\disc_localisation\disc_localisation_100226_512\disc_localisation_results.json")

# Load CSV
path = Path(__file__).parents[1]
csv = path / "optic_disc_localisation" / "ratings" / "combined_candidates" / "best_candidates_050226.csv"
df_ratings = pd.read_csv(csv)
df_ratings = df_ratings.set_index("image")


def resize_scale_factor(w: int, h: int, target_size: int) -> float:
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


def quarter_boxes(w: int, h: int, overlap_frac: float) -> dict[str, tuple]:
    """
    Four quadrant boxes for an image of size (w, h).
    overlap_frac extends each quarter beyond the midpoint — 0.0 gives exact quadrants.
    """
    mid_x = w / 2
    mid_y = h / 2
    ox = overlap_frac * mid_x
    oy = overlap_frac * mid_y

    return {
        "tl": (0,          0,          mid_x + ox, mid_y + oy),
        "tr": (mid_x - ox, 0,          w,           mid_y + oy),
        "bl": (0,          mid_y - oy, mid_x + ox, h          ),
        "br": (mid_x - ox, mid_y - oy, w,           h          ),
    }


with open(json_file, "r") as f:
    data = json.load(f)

df = pd.DataFrame.from_dict(data, orient="index")

for cls in CLASSES:
    (dest / cls).mkdir(parents=True, exist_ok=True)
    src_dir = source / cls
    dst_dir = dest / cls

    for img_path in src_dir.iterdir():
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        img_name = img_path.name
        if img_name not in df.index:
            print(f"[WARN] {cls}/{img_name} not in JSON; skipping.")
            continue

        if df.loc[img_name, "centre"] is None:
            print(f"[WARN] {cls}/{img_name}: no blob detected; skipping.")
            continue

        name_clean = Path(img_name).stem

        if df_ratings.loc[name_clean, "rating"] in REJECT_CLASS:
            print(f"[WARN] {cls}/{img_name}: disc localisation classed as poor; skipping.")
            continue

        cx_512, cy_512 = df.loc[img_name, "centre"]
        r_512 = float(df.loc[img_name, "radius"])

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            w0, h0 = img.size

            s   = resize_scale_factor(w0, h0, TARGET_SIZE)
            cx0 = float(cx_512) / s
            cy0 = float(cy_512) / s
            r0  = float(r_512)  / s

            side0 = RADIUS_SCALE_FACTOR * r0

            # Produce the same intermediate centre crop as the original script
            centre = crop_with_padding(img, centred_square_box(cx0, cy0, side0))
            centre = centre.resize((OUT_SIZE, OUT_SIZE), Image.LANCZOS)

            # Split into quarters and upsample each back to OUT_SIZE
            stem   = img_path.stem
            suffix = img_path.suffix
            for label, box in quarter_boxes(OUT_SIZE, OUT_SIZE, OVERLAP_FRAC).items():
                quarter = centre.crop(box)
                quarter = quarter.resize((OUT_SIZE, OUT_SIZE), Image.LANCZOS)
                quarter.save(dst_dir / f"{stem}_q_{label}{suffix}")