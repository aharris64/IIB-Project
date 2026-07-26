"""Basic per-channel extraction and resizing shared by all three localisation
pipelines (blob_method, vessel_method, combined_method)."""

import numpy as np
from PIL import Image

from optic_disc_localisation.image_processing.fov_processing import create_mask, inpaint


def extract_channel(img, channel):
    """
    Returns the normalized [0, 1]  channel from RGB array image in float32 array
    """
    if channel == 1: # RED
        red_img = img[:, :, 0].astype(np.float32) / 255.0
        return red_img
    elif channel == 2: # GREEN
        grn_img = img[:, :, 1].astype(np.float32) / 255.0
        return grn_img
    elif channel == 3: # BLUE
        blu_img = img[:, :, 2].astype(np.float32) / 255.0
        return blu_img
    return None


def extract_channel_masked(img, channel):
    """Extract a channel, build its FOV mask, and inpaint the dark border.

    Shared by blob_method, vessel_method, and combined_method's per-channel
    preprocessing step (previously duplicated 3-line extract/mask/inpaint sequence
    in each). Returns (channel_img, fov_mask, inpainted_img).
    """
    ch_img = extract_channel(img, channel)
    fov_mask = create_mask(ch_img)
    inpaint_img = inpaint(ch_img, fov_mask)
    return ch_img, fov_mask, inpaint_img


def resize(img, target_size):
    """
    Inputs PIL Image and returns resized image as float32 array
    Preserved aspect ratio, where the shorter side is the target size
    If all sides smaller than target size do nothing
    Faciliates faster disc detection
    """
    w, h = img.size

    if w < target_size and h < target_size:
        return np.array(img).astype(np.float32)
    if w < h:
        new_w = target_size
        new_h = int(h * target_size / w)
    else:
        new_h = target_size
        new_w = int(w * target_size / h)

    resize_img = img.resize((new_w, new_h), Image.LANCZOS) 
    # Options for resampling: NEAREST, BOX, BILINEAR, HAMMING, BICUBIC, LANCZOS
    # LANCZOS chosen as resizing performed as has high quality downsampling
    return np.array(resize_img).astype(np.float32)