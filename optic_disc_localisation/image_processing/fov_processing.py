"""Field-of-view (FOV) masking and border inpainting for fundus images."""

import numpy as np
import cv2

def create_mask(img, threshold_value=20/255, distance_to_edge=10, max_border_fraction=0.001):
    """
    Creates a masks to remove dark borders around fundus (assumes iamge is array [0,1] float)
    
    threshold_value: intensity cutoff to seperate fov and the dark border
    distance_to_edge: minimum Euclidean distance a pixel must have from the fov boundary to be included in the mask
    max_border_fraction: maximum fraction of the image that can be below threshold to assume there is no border

    Returns a binary mask where 1 indicates valid region and 0 indicates outside fov
    """
    border_mask = (img > threshold_value).astype(np.uint8)
    coverage = border_mask.mean() # Fraction of pixels above threhsold

    if coverage < (1 -  max_border_fraction):
        # Keep only the largest connected component (retina FOV)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(border_mask, connectivity=4)
        # 4 Connectivity vs 8 Connectivity (4 = Up, Down, L, R  8 = U, D, L, R, Diagonals)
        if num > 1:
            # Skip Label 0 which is the background (largest = 1 + ...)
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            border_mask = (labels == largest).astype(np.uint8)
        else:
            border_mask[:] = 1
    else:
        # If coverage is very large assume there is no border and treat the entire image as valid
        border_mask[:] = 1

    dist = cv2.distanceTransform(border_mask, distanceType=cv2.DIST_L2, maskSize=5)
    # Mask size looks in a 5×5 neighbourhood to approximate Euclidean distance by propagation

    # Keeps only pixels at least distance_to_edge pixels away from the boundary
    fov_mask = (dist >= distance_to_edge).astype(np.uint8)
    
    return fov_mask

def inpaint(img, mask, inpaint_radius=7, inpaint_method=cv2.INPAINT_TELEA):
    """
    Inpaint border using OpenCV inpaint algorithm outside using inside fov image content

    inpaint_radius: Neighborhood radius used for inpainting
    inpaint_method: OpenCV inpainting method (TELEA vs NS)
        INPAINT_TELEA default as faster and produces good enough results

    Returns inpainted image as float array in [0, 1]
    """
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8) # OpenCV only accepts 8-bit images
    out_mask = ((mask == 0).astype(np.uint8) * 255) # Creates inpaint mask (outside fov = 255, inside = 0)
    img_inp = cv2.inpaint(img, out_mask, inpaint_radius, inpaint_method).astype(np.float32) / 255.0

    return img_inp

