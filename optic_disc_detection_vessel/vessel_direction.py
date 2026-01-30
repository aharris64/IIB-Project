import numpy as np
from skimage.measure import label, regionprops
from skimage.draw import line, line_aa
import cv2

def weighted_pca_points_2d(points, thickness, thickness_weight=1.0):

    weights = thickness * thickness_weight
    
    weights_norm = weights / weights.sum()
    weighted_points = (weights_norm[:, None] * points)
    mean = weighted_points.sum(axis=0)
    centred_points = points - mean

    cov = (centred_points.T * weights_norm) @ centred_points  # 2x2 weighted covariance
    evalues, evectors = np.linalg.eigh(cov)  # ascending
    direction = evectors[:, 1]      # largest eigenvalue

    return mean, direction


def cluster_directions(skeleton_img, thickness_skel, min_area=2):

    labels = label(skeleton_img, connectivity=2)
    regions = regionprops(labels)

    results = []

    for region in regions:
        if region.area < min_area:
            continue

        # (row, col) = (y, x)
        coords_rc = region.coords              
        r = coords_rc[:, 0]
        c = coords_rc[:, 1]

        points_xy = np.column_stack([c, r])

        # Thickness sampled in image indexing (row, col)
        t_weights = thickness_skel[r, c].astype(float)

        mean, direction = weighted_pca_points_2d(points_xy, t_weights)
        weight = t_weights.sum()

        results.append([mean, direction, weight])

    total_weights = sum(r[2] for r in results)
    for r in results:
        r[2] = r[2] / total_weights   # normalize 

    return results



def pca_on_grid_boxes(skeleton_img, thickness_skel, box_size=32, min_points=5, weight_power=1.0, eps=1e-12):
    """
    Split the image into non-overlapping box_size x box_size tiles.
    For each tile, collect skeleton pixels and run weighted PCA.

    Returns a list of dicts:
      {"mean_xy":..., "direction_xy":..., "weight":..., "box_rc":(r0,r1,c0,c1), "n_points":...}
    """
    H, W = skeleton_img.shape
    results = []

    for r0 in range(0, H, box_size):
        r1 = min(r0 + box_size, H)
        for c0 in range(0, W, box_size):
            c1 = min(c0 + box_size, W)

            skel_box = skeleton_img[r0:r1, c0:c1]
            if not np.any(skel_box):
                continue

            rr, cc = np.nonzero(skel_box)  # coords inside box (row, col)
            if rr.size < min_points:
                continue

            rr_g = rr + r0
            cc_g = cc + c0

            # points in (x,y)
            points_xy = np.column_stack([cc_g, rr_g]).astype(float)

            t = thickness_skel[rr_g, cc_g].astype(float)
            t = np.clip(t, 0.0, None)

            # Per-point PCA weights
            pca_weights = np.power(t, weight_power)

            mean, direction = weighted_pca_points_2d(points_xy, pca_weights)
            if mean is None:
                continue

            # Box weight to control brightness (includes both count and thickness)
            weight = float(pca_weights.sum())

            results.append({
                "mean_xy": mean,
                "direction_xy": direction,
                "weight": weight,
                "box_rc": (r0, r1, c0, c1),
                "n_points": int(rr.size)
            })

    # Normalize weights to sum to 1 (optional; useful for consistent brightness)
    if results:
        total = sum(d["weight"] for d in results)
        if total > eps:
            for d in results:
                d["weight"] /= total

    return results

def render_direction_image(shape_hw, results, l=5, normalize=True, eps=1e-12):
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
    weights = np.array([r[2] for r in results], dtype=float)

    if normalize:
        wmax = weights.max()
        scale = 1.0 / (wmax + eps)
    else:
        scale = 1.0

    for mean, direction, w in results:

        mx, my = mean
        dx, dy = direction

        intensity = float(w * scale)

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

def render_grid_directions(shape_hw, grid_results, l=16, normalize_to_max=True, eps=1e-12):
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

def estimate_convergence_point(means, directions, weights=None, eps=1e-9):
    """
    means:      (N,2) array of m_i
    directions: (N,2) array of unit d_i (axis is fine: ±d_i gives same P_i)
    weights:    (N,)  array, optional (defaults to all ones)

    returns: p (2,) estimated intersection/convergence point
    """
    means = np.asarray(means, float)
    dirs  = np.asarray(directions, float)
    N = means.shape[0]

    if weights is None:
        w = np.ones(N, dtype=float)
    else:
        w = np.asarray(weights, float)

    I = np.eye(2)
    A = np.zeros((2, 2), dtype=float)
    b = np.zeros(2, dtype=float)

    for mi, di, wi in zip(means, dirs, w):
        n = np.linalg.norm(di)
        if n < eps or wi <= 0:
            continue
        di = di / n

        P = I - np.outer(di, di)   # projector onto normal space
        A += wi * P
        b += wi * (P @ mi)

    # Solve A p = b
    if np.linalg.cond(A) > 1/eps:
        # Ill-conditioned: fall back to least-squares
        p, *_ = np.linalg.lstsq(A, b, rcond=None)
    else:
        p = np.linalg.solve(A, b)

    return p

def ray_tmax_to_border(mx, my, dx, dy, H, W, eps=1e-12):
    """
    Ray: (x,y) = (mx,my) + t*(dx,dy), t>=0
    Returns max t until it exits the image [0,W-1]x[0,H-1].
    """
    t_candidates = []

    # x boundaries: x=0 and x=W-1
    if abs(dx) > eps:
        t = (0.0 - mx) / dx
        if t >= 0: t_candidates.append(t)
        t = ((W - 1) - mx) / dx
        if t >= 0: t_candidates.append(t)

    # y boundaries: y=0 and y=H-1
    if abs(dy) > eps:
        t = (0.0 - my) / dy
        if t >= 0: t_candidates.append(t)
        t = ((H - 1) - my) / dy
        if t >= 0: t_candidates.append(t)

    if not t_candidates:
        return 0.0

    # We need the smallest t that brings us to any border (first hit).
    tmax = min(t_candidates)

    # But ensure the intersection point is inside the rectangle (numeric safety)
    x = mx + tmax * dx
    y = my + tmax * dy
    if x < -1 or x > W or y < -1 or y > H:
        # fallback: clamp
        tmax = max(0.0, tmax)

    return float(tmax)


def vote_lines_attenuated(shape_hw, results, sigma=80.0, step=1.0,
                                normalize_weights=True, eps=1e-12):
    """
    For each (mean, direction), draw two rays from the mean to the image border
    along ±direction. Attenuate with exp(-t^2/(2 sigma^2)).
    """
    H, W = shape_hw
    acc = np.zeros((H, W), dtype=np.float32)

    if not results:
        return acc

    w = np.array([r["weight"] for r in results], dtype=np.float32)
    if normalize_weights:
        w = w / (w.sum() + eps)

    for r, wi in zip(results, w):
        if wi <= 0:
            continue

        mx, my = map(float, r["mean_xy"])        # (x,y)
        dx, dy = map(float, r["direction_xy"])   # (dx,dy)

        n = np.hypot(dx, dy)
        if n < eps:
            continue
        dx /= n
        dy /= n

        for sgn in (1.0, -1.0):
            ddx, ddy = sgn * dx, sgn * dy

            tmax = ray_tmax_to_border(mx, my, ddx, ddy, H, W, eps=eps)
            if tmax <= 0:
                continue

            ts = np.arange(0.0, tmax + step, step, dtype=np.float32)
            att = np.exp(-(ts * ts) / (2.0 * sigma * sigma)).astype(np.float32)

            xs = mx + ts * ddx
            ys = my + ts * ddy

            rr = np.rint(ys).astype(np.int32)
            cc = np.rint(xs).astype(np.int32)

            valid = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            if not np.any(valid):
                continue

            rr = rr[valid]
            cc = cc[valid]
            vv = (wi * att[valid]).astype(np.float32)

            acc[rr, cc] += vv

    return acc


def blur_and_find_peak(acc, blur_sigma=20.0):
    """
    Gaussian blur accumulator and return peak location.
    returns: blurred (H,W), p_xy (x,y), peak_value
    """
    if blur_sigma <= 0:
        blurred = acc
    else:
        k = int(6 * blur_sigma + 1)
        if k % 2 == 0:
            k += 1
        blurred = cv2.GaussianBlur(acc, (k, k), blur_sigma)

    r, c = np.unravel_index(np.argmax(blurred), blurred.shape)
    p_xy = np.array([float(c), float(r)], dtype=float)
    return blurred, p_xy, float(blurred[r, c])