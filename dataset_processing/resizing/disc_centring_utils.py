"""Shared geometry, loading, and quality-check helpers for the disc-centred resize
scripts in this folder (currently just disc_centred_resize.py) — factored out since
they were previously duplicated across several near-identical scripts."""

import json

import pandas as pd
from PIL import Image, ImageOps


def resize_scale_factor(w: int, h: int, target_size: int) -> float:
    """Scale factor mapping an image's original size to target_size on its short side."""
    if w < target_size and h < target_size:
        return 1.0
    return target_size / min(w, h)


def centred_square_box(cx: float, cy: float, side: float) -> tuple:
    """(left, top, right, bottom) box of the given side length, centred on (cx, cy)."""
    half = side / 2.0
    return (cx - half, cy - half, cx + half, cy + half)


def crop_with_padding(img: Image.Image, box: tuple, pad_color=(0, 0, 0)) -> Image.Image:
    """Crop img to box, padding with pad_color first if box extends past img's edges."""
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
            fill=pad_color,
        )
        left  += pad_left;  right  += pad_left
        top   += pad_top;   bottom += pad_top

    return img.crop((int(round(left)), int(round(top)), int(round(right)), int(round(bottom))))


def degrade_resolution(img: Image.Image, low_res: int, out_size: int) -> Image.Image:
    """Downsample img to low_res x low_res then upsample back to out_size (LANCZOS)."""
    img_low = img.resize((low_res, low_res), Image.LANCZOS)
    return img_low.resize((out_size, out_size), Image.LANCZOS)


def load_localisation_and_ratings(json_file, ratings_csv):
    """Load the ODL results JSON and manual-ratings CSV, indexed for per-image lookup.

    Returns (df_loc, df_ratings): df_loc indexed by full filename (from the JSON),
    df_ratings indexed by filename stem (the "image" column of the ratings CSV).
    """
    with open(json_file, "r") as f:
        localisation_data = json.load(f)
    df_loc = pd.DataFrame.from_dict(localisation_data, orient="index")

    df_ratings = pd.read_csv(ratings_csv)
    df_ratings = df_ratings.set_index("image")

    return df_loc, df_ratings


def check_image_quality(img_name, stem, df_loc, df_ratings, reject_class):
    """Check img_name/stem passes localisation + manual-rating checks.

    Returns (True, None) if usable, or (False, reason) for the first failed check:
    missing from the localisation JSON, no blob detected, missing from the ratings
    CSV, or manually rated poor (rating in reject_class).
    """
    if img_name not in df_loc.index:
        return False, "not in localisation JSON"

    if df_loc.loc[img_name, "centre"] is None:
        return False, "no blob detected"

    if stem not in df_ratings.index:
        return False, "not in ratings CSV"

    if df_ratings.loc[stem, "rating"] in reject_class:
        return False, "disc localisation rated poor"

    return True, None
