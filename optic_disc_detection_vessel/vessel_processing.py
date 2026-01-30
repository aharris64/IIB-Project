import numpy as np
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

def inverse_bool_img(img):
    inv_img = ~img.astype(bool)
    return inv_img

def isolate_major_vessels(img, min_area = 10):
    """
    Remove vessels with an area smaller than min_area
    """
    lab = label(img, connectivity=2)
    major_vessels = np.zeros_like(img, dtype=bool)
    for r in regionprops(lab):
        if r.area >= min_area:
            major_vessels[lab == r.label] = True

    return major_vessels

def vessel_skeleton(img):
    return skeletonize(img)

def vessel_thickness_skeleton(vessel_img, skeleton_img):
    """
    Return normalised thickness weighted skeleton
    """
    dist = distance_transform_edt(vessel_img)

    thickness_map = np.zeros_like(dist, dtype=np.float32)
    thickness_map[skeleton_img] = 2.0 * dist[skeleton_img]

    thickness = thickness_map.copy()
    thickness = thickness / (thickness.max() + 1e-8)

    return thickness