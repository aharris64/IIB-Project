import numpy as np
from skimage.draw import line, line_aa

def render_grid_directions(shape_hw, results, l=5, normalize=True, eps=1e-12):
    """
    shape_hw : (H, W)
    results  : [(mean_xy, direction_xy, weight), ...]
               mean=(x,y), direction=(dx,dy)
    l        : half-length of line
    """

    H, W = shape_hw
    out = np.zeros((H, W), dtype=np.float32)

    if len(results) == 0:
        return out

    # Collect weights
    weights = np.array([d["weight"] for d in results], float)

    if normalize:
        wmax = weights.max()
        scale = 1.0 / (wmax + eps)
    else:
        scale = 1.0

    for d in results:
        mx, my = d["mean_xy"]
        dx, dy = d["direction_xy"]
        intensity = float(d["weight"] * scale)

        # Endpoints (x,y)
        x0 = mx - l * dx
        x1 = mx + l * dx
        y0 = my - l * dy
        y1 = my + l * dy

        # Convert to (row,col)
        r0, c0 = int(round(y0)), int(round(x0))
        r1, c1 = int(round(y1)), int(round(x1))

        # Clip
        r0 = np.clip(r0, 0, H - 1)
        r1 = np.clip(r1, 0, H - 1)
        c0 = np.clip(c0, 0, W - 1)
        c1 = np.clip(c1, 0, W - 1)

        # Rasterize line
        rr, cc = line(r0, c0, r1, c1)

        # Draw with intensity (use max to avoid overwriting brighter lines)
        out[rr, cc] = np.maximum(out[rr, cc], intensity)

    return out

def render_vessel_rays(shape_hw, grid_results, l=16, normalize_to_max=True, eps=1e-12):
    """
    Render each box PCA direction as a line centered at the box mean.
    Brightness proportional to weight.
    """
    H, W = shape_hw
    out = np.zeros((H, W), dtype=np.float32)

    if not grid_results:
        return out

    weights = np.array([d["weight"] for d in grid_results], float)
    scale = 1.0 / (weights.max() + eps) if normalize_to_max else 1.0

    for d in grid_results:
        mx, my = d["mean_xy"]
        dx, dy = d["direction_xy"]
        intensity = float(d["weight"] * scale)

        x0 = mx - l * dx; x1 = mx + l * dx
        y0 = my - l * dy; y1 = my + l * dy

        r0, c0 = int(round(y0)), int(round(x0))
        r1, c1 = int(round(y1)), int(round(x1))

        r0 = np.clip(r0, 0, H - 1); r1 = np.clip(r1, 0, H - 1)
        c0 = np.clip(c0, 0, W - 1); c1 = np.clip(c1, 0, W - 1)

        rr, cc, val = line_aa(r0, c0, r1, c1)
        out[rr, cc] = np.maximum(out[rr, cc], intensity * val)

    return out