import numpy as np

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