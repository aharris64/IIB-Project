from pathlib import Path
import json
import pandas as pd
from PIL import Image, ImageOps
 
 
# ----------------------------------------------------------------------------
# ------- WILL ONLY WORK ON MANUALLY RATED DISC (I.E. 512 TARGET_SIZE) -------
# ----------------------------------------------------------------------------
# Augmentation applied to ALL images (split later).
# Per source image, 6 outputs are saved to the destination:
#
#   <stem><ext>              — original centre crop
#   <stem>_hflip<ext>        — horizontal flip of centre crop
#   <stem>_crop_tl<ext>      — offset crop: top-left
#   <stem>_crop_tr<ext>      — offset crop: top-right
#   <stem>_crop_bl<ext>      — offset crop: bottom-left
#   <stem>_crop_br<ext>      — offset crop: bottom-right
#
# When splitting later, remove augmented variants (_hflip, _crop_*) from the
# val and test sets — evaluation must be on original centre crops only.
# ----------------------------------------------------------------------------
 
CLASSES = ["normal", "papilledema", "pseudopapilledema"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
 
TARGET_SIZE = 512
OUT_SIZE = 224
PAD_COLOR = (0, 0, 0)
RADIUS_SCALE_FACTOR = 4.0
REJECT_CLASS = [4]
 
# Offset crops: shift centre by this fraction of the crop side length
OFFSET_FRAC = 0.15
 
dataset_name = f"disc_centred_r{RADIUS_SCALE_FACTOR}_cl{''.join(str(c) for c in REJECT_CLASS)}_augmented"
 
source    = Path(r"C:\Users\adam6\OneDrive\Documents\University\Engineering\Engineering IIB\IIB Project\Datasets\Processed Datasets\raw")
dest_root = Path(r"C:\Users\adam6\OneDrive\Documents\University\Engineering\Engineering IIB\IIB Project\Datasets\Processed Datasets")
dest      = dest_root / dataset_name
 
json_file = Path(r"C:\Users\adam6\OneDrive\Documents\University\Engineering\Engineering IIB\IIB Project\Datasets\Processed Datasets\disc_localisation\disc_localisation_100226_512\disc_localisation_results.json")
 
path = Path(__file__).parents[1]
csv  = path / "optic_disc_localisation" / "ratings" / "combined_candidates" / "best_candidates_050226.csv"
df_ratings = pd.read_csv(csv)
df_ratings = df_ratings.set_index("image")
 
 
# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
 
def resize_scale_factor(w: int, h: int, target_size: int) -> float:
    if w < target_size and h < target_size:
        return 1.0
    return target_size / min(w, h)
 
 
def centred_square_box(cx: float, cy: float, side: float) -> tuple:
    half = side / 2.0
    return (cx - half, cy - half, cx + half, cy + half)
 
 
def crop_with_padding(img: Image.Image, box: tuple) -> Image.Image:
    left, top, right, bottom = box
    w, h = img.size
 
    pad_left   = max(0, int(round(-left)))
    pad_top    = max(0, int(round(-top)))
    pad_right  = max(0, int(round(right - w)))
    pad_bottom = max(0, int(round(bottom - h)))
 
    if any(p > 0 for p in (pad_left, pad_top, pad_right, pad_bottom)):
        img = ImageOps.expand(
            img,
            border=(pad_left, pad_top, pad_right, pad_bottom),
            fill=PAD_COLOR,
        )
        left  += pad_left;  right  += pad_left
        top   += pad_top;   bottom += pad_top
 
    return img.crop((int(round(left)), int(round(top)), int(round(right)), int(round(bottom))))
 
 
# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------
 
def make_centre_crop(img: Image.Image, cx: float, cy: float, side: float) -> Image.Image:
    """Centred square crop resized to OUT_SIZE."""
    return crop_with_padding(img, centred_square_box(cx, cy, side)).resize(
        (OUT_SIZE, OUT_SIZE), Image.LANCZOS
    )
 
 
def make_offset_crops(
    img: Image.Image,
    cx: float,
    cy: float,
    side: float,
) -> dict[str, Image.Image]:
    """
    Four crops offset by ±OFFSET_FRAC * side in x and y.
    Keys: tl (top-left), tr (top-right), bl (bottom-left), br (bottom-right).
    """
    delta = OFFSET_FRAC * side
    offsets = {
        "tl": (-delta, -delta),
        "tr": ( delta, -delta),
        "bl": (-delta,  delta),
        "br": ( delta,  delta),
    }
    return {
        label: crop_with_padding(
            img, centred_square_box(cx + dx, cy + dy, side)
        ).resize((OUT_SIZE, OUT_SIZE), Image.LANCZOS)
        for label, (dx, dy) in offsets.items()
    }
 
 
# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------
 
with open(json_file, "r") as f:
    localisation_data = json.load(f)
 
df_loc = pd.DataFrame.from_dict(localisation_data, orient="index")
 
saved = skipped = 0
 
for cls in CLASSES:
    src_dir = source / cls
    dst_dir = dest / cls
    dst_dir.mkdir(parents=True, exist_ok=True)
 
    if not src_dir.exists():
        print(f"[INFO] Source directory not found, skipping: {src_dir}")
        continue
 
    for img_path in sorted(src_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
 
        img_name = img_path.name
        stem     = img_path.stem
        suffix   = img_path.suffix
 
        # --- Localisation check ---
        if img_name not in df_loc.index:
            print(f"[WARN] {cls}/{img_name}: not in localisation JSON; skipping.")
            skipped += 1
            continue
 
        if df_loc.loc[img_name, "centre"] is None:
            print(f"[WARN] {cls}/{img_name}: no blob detected; skipping.")
            skipped += 1
            continue
 
        # --- Quality rating check ---
        if stem not in df_ratings.index:
            print(f"[WARN] {cls}/{img_name}: not in ratings CSV; skipping.")
            skipped += 1
            continue
 
        if df_ratings.loc[stem, "rating"] in REJECT_CLASS:
            print(f"[WARN] {cls}/{img_name}: disc localisation rated poor; skipping.")
            skipped += 1
            continue
 
        # --- Map localisation coords from 512-space -> original image space ---
        cx_512, cy_512 = df_loc.loc[img_name, "centre"]
        r_512  = float(df_loc.loc[img_name, "radius"])
 
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            w0, h0 = img.size
 
            s     = resize_scale_factor(w0, h0, TARGET_SIZE)
            cx0   = float(cx_512) / s
            cy0   = float(cy_512) / s
            r0    = float(r_512)  / s
            side0 = RADIUS_SCALE_FACTOR * r0
 
            # 1. Original centre crop
            centre = make_centre_crop(img, cx0, cy0, side0)
            centre.save(dst_dir / f"{stem}{suffix}")
 
            # 2. Horizontal flip of centre crop
            centre.transpose(Image.FLIP_LEFT_RIGHT).save(
                dst_dir / f"{stem}_hflip{suffix}"
            )
 
            # 3. Four offset crops
            for label, crop in make_offset_crops(img, cx0, cy0, side0).items():
                crop.save(dst_dir / f"{stem}_crop_{label}{suffix}")
 
            saved += 1
 
print(f"\nDone. {saved} source images processed → {saved * 6} total outputs. {skipped} skipped.")
print(f"Output: {dest}")
print(
    "\nNOTE: when splitting, strip augmented variants (_hflip, _crop_*) from "
    "val and test — evaluate on original centre crops only."
)