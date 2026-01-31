import numpy as np
import cv2

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


def generate_vessel_rays(shape_hw, results, sigma=80.0, step=1.0,
                         normalize_weights=True, use_weights=True,
                         eps=1e-12):
    """
    For each (mean, direction), draw two rays from the mean to the image border
    along ±direction. Attenuate with exp(-t^2/(2 sigma^2)).

    If use_weights=False, all rays contribute equally.
    """
    H, W = shape_hw
    acc = np.zeros((H, W), dtype=np.float32)

    if not results:
        return acc

    # --- Weight handling ---
    if use_weights:
        w = np.array([r.get("weight", 1.0) for r in results],
                     dtype=np.float32)

        if normalize_weights:
            w = w / (w.sum() + eps)
    else:
        # All rays = 1
        w = np.ones(len(results), dtype=np.float32)

        if normalize_weights:
            w = w / (w.sum() + eps)
    # -----------------------

    for r, wi in zip(results, w):
        if wi <= 0:
            continue

        mx, my = map(float, r["mean_xy"])        # (x,y)
        dx, dy = map(float, r["direction_xy"])  # (dx,dy)

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


def blur_vessel_rays(acc, blur_sigma=20.0):
    if blur_sigma <= 0:
        blurred = acc
    else:
        k = int(6 * blur_sigma + 1)
        if k % 2 == 0:
            k += 1
        blurred = cv2.GaussianBlur(acc, (k, k), blur_sigma)

    return blurred

def find_convergence_point(blurred):
    r, c = np.unravel_index(np.argmax(blurred), blurred.shape)
    p_xy = np.array([float(c), float(r)], dtype=float)
    return p_xy, float(blurred[r, c])
    