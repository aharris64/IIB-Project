import numpy as np
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from skimage.filters import threshold_multiotsu
import cv2

def otsu_thresholding(img, level=0):
    """
    Apply Multi Otsu thresholding at level 0 or 1
    Returns binary image
    """
    thresh_multi_ostu = threshold_multiotsu(img)
    multi_ostu_r1 = img > thresh_multi_ostu[level]

    return multi_ostu_r1

def inverse_bool_img(img):
    """
    Invert boolean image
    """
    inv_img = ~img.astype(bool)
    return inv_img

def vessel_inpaint(img, vessel_mask):
    """
    Inpaint image provided with the vessel mask to remove vessels
    """

    img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8) # OpenCV only accepts 8-bit images
    mask_u8 = (vessel_mask > 0).astype(np.uint8) * 255

    img_inp_u8 = cv2.inpaint(img_u8, mask_u8, 5, cv2.INPAINT_NS).astype(np.float32) / 255.0

    return img_inp_u8

def isolate_major_vessels(img, min_area=10):
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