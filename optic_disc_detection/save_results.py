import numpy as np
import os
import cv2

def save_image(img, save_folder, name):
    """
    Save an image array, handling grayscale (H×W) and RGB (H×W×3) images
    """
    
    if img.ndim == 2:
        # Grayscale image (assumed float [0,1])
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        save_path = os.path.join(save_folder, name)
        cv2.imwrite(save_path, img)

    elif img.ndim == 3 and img.shape[2] == 3:
        # RGB image
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        save_path = os.path.join(save_folder, name)
        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    else:
        raise ValueError("Unsupported image shape")
    
def save_mask_overlay(img, mask, save_folder, name, alpha=0.35):
    """
    Save an RGB visualization overlaying the excluded mask region in red
    """
    # Ensure correct dimensions and channels
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Create overlay where mask is 0 (border)
    overlay = img.copy()
    overlay[mask == 0] = (0, 0, 255) # Colour overlay red
    mask_overlay = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    save_path = os.path.join(save_folder, name)
    cv2.imwrite(save_path, mask_overlay)

def save_candidate_overlay(img, candidates, save_folder, name):
    """
    Overlay candidates on img provided
    """

    if img.ndim == 2:
        # Grayscale image (assumed float [0,1])
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        img_c = img.copy()

    elif img.ndim == 3 and img.shape[2] == 3:
        # RGB image
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        img_c = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    else:
        raise ValueError("Unsupported image shape")

    for (centre, radius, resp) in candidates:
        cv2.circle(img_c, centre, int(round(radius)), (0, 255, 0), 1)
        cv2.circle(img_c, centre, 2, (0, 0, 255), -1)

    save_path = os.path.join(save_folder, name)
    cv2.imwrite(save_path, img_c)