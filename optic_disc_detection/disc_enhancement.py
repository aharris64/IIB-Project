import cv2
import numpy as np

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

def histogram_equalization(img):
    """
    Apply Contrast Histogram Equalization
    """
    img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img_equ = cv2.equalizeHist(img_u8)
    img_equ = img_equ.astype(np.float32) / 255.0

    return img_equ.astype(np.float32)

def clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    """
    img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    img_clahe = clahe.apply(img_u8)
    img_clahe = img_clahe.astype(np.float32) / 255.0

    return img_clahe.astype(np.float32)

def gamma_brightening(img, gamma=10):
    """
    Bright-biased contrast enhancement
    """
    img_gamma = img ** gamma
    return img_gamma.astype(np.float32)

def percentage_based_enhancement(img, p_low=90, p_high=99.5):
    """
    Linearly stretch intensities between low and high intensities calculated by 
    percentiles p_low and p_high
    Pixels below low intensity become < 0, above high intensity become > 1
    """
    # Compute intensity values corresponding to the chosen percentiles
    low_int = np.percentile(img, p_low)
    high_int = np.percentile(img, p_high)

    img_enh = (img - low_int) / (high_int - low_int + 1e-6)
    img_enh = np.clip(img_enh, 0, 1)

    return img_enh.astype(np.float32)

def percentile_controlled_gamma(img, pers=(95, 97.5), targets=(0.1, 0.8)):
    """
    This method computes an intensity mapping of the form y = A * x^gamma clipped at 1
    such that two specified input percentiles are mapped to desired target
    output intensities
    """
    # Compute intensity values corresponding to the chosen percentiles
    low_int = np.percentile(img, pers[0])
    high_int = np.percentile(img, pers[1])

    gamma = np.log(targets[0]/targets[1])/np.log(low_int/high_int)
    A = targets[1] / (low_int ** gamma)

    img_enh = A * (img ** gamma)
    img_enh = np.clip(img_enh, 0, 1)

    return img_enh.astype(np.float32)