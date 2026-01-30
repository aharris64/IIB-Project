from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt
import numpy as np

def vessel_clusters_and_dist(skel_img, num_clusters=5, min_area=30):

    # TODO: Check binary image

    distance = np.zeros_like(skel_img, dtype=np.float32)
    labels = label(skel_img, connectivity=2)
    regions = regionprops(labels)

    regions_img = np.zeros_like(skel_img, dtype=bool)
    regions = [r for r in regions if r.area >= min_area]
    regions = sorted(regions, key=lambda r: r.area, reverse=True)[:num_clusters]

    for r in regions:
        # Create binary mask for a single cluster
        regions_img[labels == r.label] = True

        coords = r.coords
        m = np.zeros_like(skel_img, dtype=bool)
        m[coords[:, 0], coords[:, 1]] = True
        
        distance += distance_transform_edt(~m).astype(np.float32)
        
    cluster_count = len(regions)
    mean_dist = distance / cluster_count
    mean_dist_norm = (mean_dist - mean_dist.min()) / (mean_dist.max() - mean_dist.min() + 1e-8)

    return mean_dist_norm, regions_img, cluster_count





        
