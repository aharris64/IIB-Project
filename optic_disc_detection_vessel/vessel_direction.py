import numpy as np
from skimage.measure import label, regionprops


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
        print(r[0], r[1], r[2])

    return results

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

