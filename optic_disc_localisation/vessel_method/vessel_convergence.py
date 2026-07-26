"""Projects each grid box's principal vessel direction (from vessel_directions.py) as
attenuated rays to the image border; where the rays converge (after blurring) is a
proxy for the optic disc / macula-adjacent point."""

import numpy as np
from optic_disc_localisation.image_processing.gaussian_processing import gaussian_blur

def ray_distance_to_border(pos_x, pos_y, dx, dy, height, width):
    """
    Ray: (x,y) = (mx,my) + t*(dx,dy), t>=0
    Returns max t until it exits the image [0,W-1]x[0,H-1].
    """
    candidates = []
    eps = 1e-12

    # If dx = 0, the ray is vertical and will never hit vertical border
    if abs(dx) > eps:
        # X boundaries: x = 0 and x = width - 1
        t = (0.0 - pos_x) / dx
        if t >= 0:
            candidates.append(t)
        
        t = ((width - 1) - pos_x) / dx
        if t >= 0:
            candidates.append(t)

    # If dy = 0, the ray is horizontal and will never hit horizontal border
    if abs(dy) > eps:
        # Y boundaries: y = 0 and y = height - 1
        t = (0.0 - pos_y) / dy
        if t >= 0:
            candidates.append(t)

        t = ((height - 1) - pos_y) / dy
        if t >= 0:
            candidates.append(t)

    # Smallest t that brings us to any border (first hit).
    tmax = min(candidates)

    return float(tmax)


def generate_vessel_rays(img, results, sigma=200, use_weights=False):
    """
    For each (mean, direction), draw two rays from the mean to the image border along ±direction
    Attenuate with gaussian: exp(-t^2/(2 sigma^2))

    If use_weights=False, all rays contribute equally, else higher weights contribute more
    """
    img_height, img_width = img.shape
    rays_img = np.zeros((img_height, img_width), dtype=np.float32)

    # Weight handling
    if use_weights:
        weights = np.array([r["weight"] for r in results], dtype=np.float32)
    else:
        weights = np.ones(len(results), dtype=np.float32)

    for ray, weight in zip(results, weights):

        # Safety check for non positive weights
        if weight <= 0:
            continue

        pos_x, pos_y = map(float, ray["mean_xy"]) # (x,y)
        dx, dy = map(float, ray["direction_xy"]) # (dx,dy)

        # Normalise direciton
        n = np.hypot(dx, dy)
        if n < 1e-12:
            continue
        dx = dx / n
        dy = dy / n

        for sign in (1.0, -1.0): # Rays to go in both direction
            ddx, ddy = sign * dx, sign * dy

            dist_to_border = ray_distance_to_border(pos_x, pos_y, ddx, ddy, img_height, img_width)
            if dist_to_border <= 0:
                continue

            sample_points = np.arange(0.0, dist_to_border + 1.0, 1.0, dtype=np.float32)
            attenuate = np.exp(-(sample_points * sample_points) / (2.0 * sigma * sigma)).astype(np.float32)

            # Compute integer pixels
            row = np.rint(pos_y + sample_points * ddy).astype(np.int32)
            col = np.rint(pos_x + sample_points * ddx).astype(np.int32)

            # Remove out of bounds pixels
            valid = (row >= 0) & (row < img_height) & (col >= 0) & (col < img_width)
            if not np.any(valid):
                continue

            # Keep valid pixels
            row = row[valid]
            col = col[valid]
            attendated_ray = (weight * attenuate[valid]).astype(np.float32)

            rays_img[row, col] += attendated_ray

    rays_img = rays_img / rays_img.max() # Normalise

    return rays_img

def blur_rays(rays_img, sigma=15):
    """
    Blur rays with gaussian of standard deviation sigma and normalise
    """
    blurred_img = gaussian_blur(rays_img, sigma=sigma) # Blur

    blurred_img = blurred_img / blurred_img.max() # Normalise

    return blurred_img

def find_convergence_point(blurred):
    """
    Finds the pixel location with the maximum intensity
    Returns with (x, y) coordinates
    """
    row, col = np.unravel_index(np.argmax(blurred), blurred.shape)

    p_xy = np.array([col, row], dtype=np.int32) # (x, y) coords

    return p_xy