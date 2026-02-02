import numpy as np

def weighted_pca_points_2d(points, thickness, thickness_weight=1.0):
    """
    Compute the weighted principal direction of a set of 2D points using PCA.

    Each point is assigned a weight based on its associated thickness value.

    The principal direction corresponds to the eigenvector associated with
    the largest eigenvalue of the weighted covariance matrix.
    """


    weights = np.power(thickness, thickness_weight)
    
    weights_norm = weights / weights.sum()
    weighted_points = (weights_norm[:, None] * points)
    mean = weighted_points.sum(axis=0)
    centred_points = points - mean

    cov = (centred_points.T * weights_norm) @ centred_points  # 2x2 weighted covariance
    evalues, evectors = np.linalg.eigh(cov)  # ascending
    direction = evectors[:, 1] # largest eigenvalue

    return mean, direction

def pca_on_grid_boxes(skeleton_img, thickness_skel, box_size=10, min_points=5, weight_power=1.0):
    """
    Split the image into non-overlapping box_size x box_size tiles.
    For each tile, collect skeleton pixels and run weighted PCA.

    Returns a list of dicts:
      {"mean_xy":..., "direction_xy":..., "weight":...}
    """
    img_height, img_width = skeleton_img.shape
    results = []

    for box_top in range(0, img_height, box_size):
        box_bottom = min(box_top + box_size, img_height)

        for box_left in range(0, img_width, box_size):
            box_right = min(box_left + box_size, img_width)

            box = skeleton_img[box_top:box_bottom, box_left:box_right]

            # If box is empty: continue
            if not np.any(box):
                continue

            box_row, box_col = np.nonzero(box)  # coords inside box (row, col)
            
            # If box has less nonzero pixels than min_points continue
            if box_row.size < min_points:
                continue

            # Get row and col coords in original image
            img_row = box_row + box_top
            img_col = box_col + box_left

            # All non_zero points in (x,y)
            points_xy = np.column_stack([img_col, img_row]).astype(float)

            # Get thickness weights
            pca_weights = thickness_skel[img_row, img_col].astype(float)

            mean, direction = weighted_pca_points_2d(points_xy, pca_weights)

            # Box weight to control brightness (includes both count and thickness)
            box_weight = float(pca_weights.sum())

            results.append({
                "mean_xy": mean,
                "direction_xy": direction,
                "weight": box_weight,
            })

    # Normalize weights to sum to 1
    total = sum(d["weight"] for d in results)
    for d in results:
        d["weight"] /= total

    return results