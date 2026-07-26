"""Debug/visualisation image saving shared by all three localisation pipelines."""

import numpy as np
import os
import cv2
from skimage.draw import line


def _prepare_bgr(img):
    """Convert a grayscale (float [0,1] or uint8) or RGB (float or uint8) image to a
    uint8 BGR array ready for cv2 drawing/imwrite. Raises on any other shape."""
    if img.ndim == 2:
        # Grayscale image (assumed float [0,1])
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        return img.copy()

    elif img.ndim == 3 and img.shape[2] == 3:
        # RGB image
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    else:
        raise ValueError("Unsupported image shape")


def save_image(img, save_folder, name):
    """
    Save an image array, handling grayscale (H×W) and RGB (H×W×3) images
    """
    img_bgr = _prepare_bgr(img)
    save_path = os.path.join(save_folder, name)
    cv2.imwrite(save_path, img_bgr)


def save_mask_overlay(img, mask, save_folder, name, alpha=0.35):
    """
    Save an RGB visualization overlaying the excluded mask region in red
    """
    # Ensure correct dimensions and channels
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Create overlay where mask is 0 (border)
    overlay = img.copy()
    overlay[mask == 0] = (0, 0, 255) # Colour overlay red
    mask_overlay = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    save_path = os.path.join(save_folder, name)
    cv2.imwrite(save_path, mask_overlay)

def save_candidate_overlay(img, candidates, save_folder, name):
    """
    Overlay candidates on img provided
    """
    img_c = _prepare_bgr(img)

    for (centre, radius, response) in candidates:
        cv2.circle(img_c, centre, int(round(radius)), (0, 255, 0), 1)
        cv2.circle(img_c, centre, 2, (0, 0, 255), -1)

    save_path = os.path.join(save_folder, name)
    cv2.imwrite(save_path, img_c)

def save_centre_overlay(img, centre, save_folder, name):
    """
    Overlay centre on image provided
    """
    img_c = _prepare_bgr(img)

    cv2.circle(img_c, centre, 2, (255, 0, 0), -1)

    save_path = os.path.join(save_folder, name)
    cv2.imwrite(save_path, img_c)

def save_vessel_centre_and_blob_candidate(img, blob_candidate, vessel_centre, save_folder, name):
    """
    Overlay both the centre and radius of blob candidate and vessel centre on the same image
    """
    img_c = _prepare_bgr(img)

    blob_centre, blob_radius, blob_resp = blob_candidate
    
    cv2.circle(img_c, blob_centre, int(round(blob_radius)), (0, 255, 0), 1)
    cv2.circle(img_c, blob_centre, 2, (0, 0, 255), -1)

    cv2.circle(img_c, vessel_centre, 2, (255, 0, 0), -1)

    save_path = os.path.join(save_folder, name)
    cv2.imwrite(save_path, img_c)

def save_grid_pca(img, results, save_path, name, l=5):
    """
    Visualises short line segments representing PCA directions at given points, 
    with brightness proportional to their weights
    """

    height, width = img.shape
    output_img = np.zeros((height, width), dtype=np.float32)

    weights = np.array([d["weight"] for d in results], float)

    for d in results:
        mx, my = d["mean_xy"]
        dx, dy = d["direction_xy"]
        intensity = float(d["weight"] / weights.max())

        # Endpoints (x,y)
        x0 = mx - l * dx
        x1 = mx + l * dx
        y0 = my - l * dy
        y1 = my + l * dy

        # Convert to (row,col)
        r0, c0 = int(round(y0)), int(round(x0))
        r1, c1 = int(round(y1)), int(round(x1))

        # Clip to image bounds
        r0 = np.clip(r0, 0, height - 1)
        r1 = np.clip(r1, 0, height - 1)
        c0 = np.clip(c0, 0, width - 1)
        c1 = np.clip(c1, 0, width - 1)

        # Rasterize line
        rr, cc = line(r0, c0, r1, c1)

        # Draw with intensity (use max to avoid overwriting brighter lines)
        output_img[rr, cc] = np.maximum(output_img[rr, cc], intensity)

    save_image(output_img, save_path, name)