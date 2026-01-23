from PIL import Image
import os
from pathlib import Path
import cv2
import numpy as np

from border_processing import create_mask, inpaint
from initial_processing import extract_channel
from save_results import save_image, save_mask_overlay
from vessel_enhancement import gaussian_blur, percentage_based_enhancement, clahe, multi_clahe, percentile_controlled_gamma

from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt

current_path = Path(__file__).resolve().parent
test_images = os.path.join(current_path, "test_images")
process_results_folder = os.path.join(current_path, "process_results")
results_folder = os.path.join(current_path, "results")
pyramid_folder = os.path.join(current_path, "pyramid")
candidate_folder = os.path.join(current_path, "candidates")

def detect_disc_channel(img, channel):

    ch_img = extract_channel(img, channel)

    # Mask to remove dark border
    fov_mask = create_mask(ch_img)
    
    # Fill the border with OpenCV inpaint algorithm
    inpaint_img = inpaint(ch_img, fov_mask)


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    vessels_open = cv2.morphologyEx(inpaint_img, cv2.MORPH_OPEN, kernel)
    blur = gaussian_blur(vessels_open, sigma=2)
    inpaint_img = blur

    # Vessel Enhancement
    clahe_img = clahe(inpaint_img)
    clahe2_img = multi_clahe(inpaint_img, 2)
    clahe3_img = multi_clahe(inpaint_img, 3)

    # Vessel Enhancement
    per_clahe_img = percentage_based_enhancement(clahe3_img)

    B = per_clahe_img.astype(bool)
    inv_img = ~B

    skeleton_img = skeletonize(inv_img)
    dist = distance_transform_edt(inv_img)
    thickness_map = np.zeros_like(dist, dtype=np.float32)
    thickness_map[skeleton_img] = 2.0 * dist[skeleton_img]
    thickness = thickness_map.copy()
    thickness = thickness / thickness.max()

    save_image(skeleton_img, process_results_folder, "6_skeleton_img" + "_ch" + str(channel) + ".png")
    save_image(dist, process_results_folder, "6_dist" + "_ch" + str(channel) + ".png")
    save_image(thickness, process_results_folder, "6_thickness" + "_ch" + str(channel) + ".png")


    vessels_open = cv2.morphologyEx(inpaint_img, cv2.MORPH_OPEN, kernel)
    vessels_close = cv2.morphologyEx(inpaint_img, cv2.MORPH_CLOSE, kernel)
    
    save_image(vessels_open, process_results_folder, "6_vessels_open" + "_ch" + str(channel) + ".png")
    save_image(vessels_close, process_results_folder, "6_vessels_close" + "_ch" + str(channel) + ".png")


    save_image(img, process_results_folder, "1_img" + "_ch" + str(channel) + ".png")
    save_image(ch_img, process_results_folder, "2_ch_img" + "_ch" + str(channel) + ".png")
    save_mask_overlay(ch_img, fov_mask, process_results_folder, "3_mask_overlay" + "_ch" + str(channel) + ".png")
    save_image(inpaint_img, process_results_folder, "4_inpaint_img" + "_ch" + str(channel) + ".png")
        
    save_image(clahe_img, process_results_folder, "5_clahe_img" + "_ch" + str(channel) + ".png")
    save_image(clahe2_img, process_results_folder, "5_clahe2_img" + "_ch" + str(channel) + ".png")
    save_image(clahe3_img, process_results_folder, "5_clahe3_img" + "_ch" + str(channel) + ".png")
    
    save_image(per_clahe_img, process_results_folder, "5_per_clahe_img" + "_ch" + str(channel) + ".png")


def detect_disc(img):
        
    detect_disc_channel(img, 2)
