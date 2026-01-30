import cv2
import numpy as np
from skimage.filters import threshold_multiotsu


def gaussian_division(img, sigma=60):
    """
    Divide the image by a gaussian blur of standard deviation sigma
    """
    blur = cv2.GaussianBlur(img, (0, 0), sigma) + 1e-6
    # (0, 0) - compute kernel size automatically as sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8

    img_corr = img / blur
    img_corr = cv2.normalize(img_corr, None, 0, 1, cv2.NORM_MINMAX)

    return img_corr.astype(np.float32)

def gaussian_subtraction(img, sigma=60):
    """
    Subtract the image by a gaussian blur of standard deviation sigma
    """
    blur = cv2.GaussianBlur(img, (0, 0), sigma) + 1e-6
    # (0, 0) - compute kernel size automatically as sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8

    img_sub = img - blur
    img_sub = cv2.normalize(img_sub, None, 0, 1, cv2.NORM_MINMAX)
    
    return img_sub.astype(np.float32)

def gaussian_blur(img, sigma=5):
    blur = cv2.GaussianBlur(img, (0, 0), sigma) + 1e-6
    # (0, 0) - compute kernel size automatically as sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
    
    return blur.astype(np.float32)

def otsu_thresholding(img):
    thresh_multi_ostu = threshold_multiotsu(img)
    multi_ostu_r1 = img > thresh_multi_ostu[0]

    return multi_ostu_r1
