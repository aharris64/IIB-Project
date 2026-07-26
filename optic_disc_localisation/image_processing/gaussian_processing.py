"""Gaussian-blur-based background/vessel enhancement helpers."""

import cv2
import numpy as np

def gaussian_subtraction(img, sigma):
    """
    Subtract the image by a gaussian blur of standard deviation sigma
    """
    # Ensure float
    img = img.astype(np.float32)

    # Remove any existing NaN/Inf
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    # Large-scale background
    blur = cv2.GaussianBlur(img, (0, 0), sigma)

    img_sub = img - blur
    img_sub -= img_sub.min()

    img_sub /= (img_sub.max() + 1e-12)

    return img_sub

def gaussian_blur(img, sigma):
    """
    Gaussian blur of standard deviation sigma
    """
    blur = cv2.GaussianBlur(img, (0, 0), sigma) + 1e-6
    # (0, 0) - compute kernel size automatically as sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
    
    return blur.astype(np.float32)

