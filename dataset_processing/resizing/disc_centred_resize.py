"""Disc-centred crop of the raw dataset, filtered by manual disc-localisation quality
rating (WILL ONLY WORK ON MANUALLY RATED DISC, i.e. 512 TARGET_SIZE) 
The four combinations used so far, and the exact dataset folder name
each produces:

  AUGMENTATION_MODE="none",        LOW_RES=None -> disc_centred_r4.0_cl34
  AUGMENTATION_MODE="flip_offset", LOW_RES=None -> disc_centred_r4.0_cl34_augmented
  AUGMENTATION_MODE="flip_offset", LOW_RES=14   -> disc_centred_r4.0_cl34_augmented_lowres14
  AUGMENTATION_MODE="quarters",    LOW_RES=None -> disc_centred_r4.0_cl34_quarters_overlap0.0

"flip_offset" saves 6 crops per source image: the centre crop, its horizontal flip, and
4 crops offset by OFFSET_FRAC in each diagonal direction (all independently re-cropped
from the original image, not from the centre crop). "quarters" instead splits the
already-resized 224x224 centre crop into 4 overlapping quadrants (overlap set by
OVERLAP_FRAC) and upsamples each back to 224x224.

LOW_RES, if set, downsamples every final saved crop to LOW_RES then back up to OUT_SIZE 
(applied per-crop, so e.g. the hflip inherits the same degradation as the centre crop it's 
derived from, while each offset/quarter crop is degraded independently after its own resize).

When splitting a dataset produced in "flip_offset" or "quarters" mode into train/val/
test, strip the augmented variants (_hflip, _crop_*, _q_*) from val and test — evaluate
on original centre crops only (see dataset_processing/train_val_test.py).
"""

import os
from pathlib import Path

from PIL import Image

from disc_centring_utils import (
    resize_scale_factor,
    centred_square_box,
    crop_with_padding,
    degrade_resolution,
    load_localisation_and_ratings,
    check_image_quality,
)

CLASSES = ["normal", "papilledema", "pseudopapilledema"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

TARGET_SIZE = 512
OUT_SIZE = 224
PAD_COLOR = (0, 0, 0)
RADIUS_SCALE_FACTOR = 4.0
REJECT_CLASS = [3, 4]

# ---- Config: choose ONE augmentation mode ----
AUGMENTATION_MODE = "none"   # "none" | "flip_offset" | "quarters"
OFFSET_FRAC = 0.15           # used only when AUGMENTATION_MODE == "flip_offset"
OVERLAP_FRAC = 0.0           # used only when AUGMENTATION_MODE == "quarters"
LOW_RES = None               # e.g. 14 to degrade every saved crop; None = no degradation

_OUTPUTS_PER_IMAGE = {"none": 1, "flip_offset": 6, "quarters": 4}


def _maybe_degrade(crop_img):
    return degrade_resolution(crop_img, LOW_RES, OUT_SIZE) if LOW_RES is not None else crop_img

# ---- Dataset name (mirrors the folder names the old separate scripts produced) ----
name = f"disc_centred_r{RADIUS_SCALE_FACTOR}_cl{''.join(str(c) for c in REJECT_CLASS)}"
if AUGMENTATION_MODE == "flip_offset":
    name += "_augmented"
elif AUGMENTATION_MODE == "quarters":
    name += f"_quarters_overlap{OVERLAP_FRAC}"
if LOW_RES is not None:
    name += f"_lowres{LOW_RES}"

DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "./Datasets")
source    = Path(DATASETS_ROOT) / "Processed Datasets" / "raw"
dest_root = Path(DATASETS_ROOT) / "Processed Datasets"
dest      = dest_root / name

json_file = Path(DATASETS_ROOT) / "Processed Datasets" / "disc_localisation" / "disc_localisation_100226_512" / "disc_localisation_results.json"
ratings_csv = Path(__file__).resolve().parents[2] / "optic_disc_localisation" / "ratings" / "data" / "combined_candidates" / "best_candidates_050226.csv"

df_loc, df_ratings = load_localisation_and_ratings(json_file, ratings_csv)

saved = skipped = 0

print(f"Mode: {AUGMENTATION_MODE}" + (f", low-res {LOW_RES}" if LOW_RES is not None else ""))
print(f"Output directory: {dest}")

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

        ok, reason = check_image_quality(img_name, stem, df_loc, df_ratings, REJECT_CLASS)
        if not ok:
            print(f"[WARN] {cls}/{img_name}: {reason}; skipping.")
            skipped += 1
            continue

        cx_512, cy_512 = df_loc.loc[img_name, "centre"]
        r_512 = float(df_loc.loc[img_name, "radius"])

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            w0, h0 = img.size

            s     = resize_scale_factor(w0, h0, TARGET_SIZE)
            cx0   = float(cx_512) / s
            cy0   = float(cy_512) / s
            r0    = float(r_512)  / s
            side0 = RADIUS_SCALE_FACTOR * r0

            centre_raw = crop_with_padding(img, centred_square_box(cx0, cy0, side0), PAD_COLOR)
            centre_raw = centre_raw.resize((OUT_SIZE, OUT_SIZE), Image.LANCZOS)

            if AUGMENTATION_MODE == "none":
                _maybe_degrade(centre_raw).save(dst_dir / f"{stem}{suffix}")

            elif AUGMENTATION_MODE == "flip_offset":
                centre_final = _maybe_degrade(centre_raw)
                centre_final.save(dst_dir / f"{stem}{suffix}")
                centre_final.transpose(Image.FLIP_LEFT_RIGHT).save(dst_dir / f"{stem}_hflip{suffix}")

                delta = OFFSET_FRAC * side0
                offsets = {
                    "tl": (-delta, -delta),
                    "tr": ( delta, -delta),
                    "bl": (-delta,  delta),
                    "br": ( delta,  delta),
                }
                for label, (dx, dy) in offsets.items():
                    crop = crop_with_padding(img, centred_square_box(cx0 + dx, cy0 + dy, side0), PAD_COLOR)
                    crop = crop.resize((OUT_SIZE, OUT_SIZE), Image.LANCZOS)
                    _maybe_degrade(crop).save(dst_dir / f"{stem}_crop_{label}{suffix}")

            elif AUGMENTATION_MODE == "quarters":
                mid = OUT_SIZE / 2
                ox = oy = OVERLAP_FRAC * mid
                quarter_boxes = {
                    "tl": (0,      0,      mid + ox, mid + oy),
                    "tr": (mid - ox, 0,      OUT_SIZE, mid + oy),
                    "bl": (0,      mid - oy, mid + ox, OUT_SIZE),
                    "br": (mid - ox, mid - oy, OUT_SIZE, OUT_SIZE),
                }
                for label, box in quarter_boxes.items():
                    quarter = centre_raw.crop(box).resize((OUT_SIZE, OUT_SIZE), Image.LANCZOS)
                    _maybe_degrade(quarter).save(dst_dir / f"{stem}_q_{label}{suffix}")

            saved += 1

outputs_per_image = _OUTPUTS_PER_IMAGE[AUGMENTATION_MODE]
print(f"\nDone. {saved} source images processed -> {saved * outputs_per_image} total outputs. {skipped} skipped.")
print(f"Output: {dest}")
if AUGMENTATION_MODE in ("flip_offset", "quarters"):
    print(
        "\nNOTE: when splitting, strip augmented variants (_hflip, _crop_*, _q_*) from "
        "val and test — evaluate on original centre crops only."
    )
