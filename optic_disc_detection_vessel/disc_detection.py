from PIL import Image
import os
from pathlib import Path
import cv2
import numpy as np

from border_processing import create_mask, inpaint
from initial_processing import extract_channel
from save_results import save_image, save_mask_overlay
from vessel_enhancement import gaussian_blur, percentage_based_enhancement, clahe, multi_clahe, percentile_controlled_gamma, gaussian_division, gaussian_subtraction

from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from skimage.filters import threshold_otsu, threshold_multiotsu
from skimage.measure import label, regionprops

current_path = Path(__file__).resolve().parent
test_images = os.path.join(current_path, "test_images")
process_results_folder = os.path.join(current_path, "process_results")
results_folder = os.path.join(current_path, "results")
pyramid_folder = os.path.join(current_path, "pyramid")
candidate_folder = os.path.join(current_path, "candidates")

def detect_disc_channel(img, channel):

    save_image(img, process_results_folder, "1_img" + "_ch" + str(channel) + ".png")

    ch_img = extract_channel(img, channel)
    save_image(ch_img, process_results_folder, "2_ch_img" + "_ch" + str(channel) + ".png")

    # Mask to remove dark border
    fov_mask = create_mask(ch_img)
    save_mask_overlay(ch_img, fov_mask, process_results_folder, "3_mask_overlay" + "_ch" + str(channel) + ".png")

    # Fill the border with OpenCV inpaint algorithm
    inpaint_img = inpaint(ch_img, fov_mask)
    save_image(inpaint_img, process_results_folder, "4_inpaint_img" + "_ch" + str(channel) + ".png")

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    # vessels_open = cv2.morphologyEx(inpaint_img, cv2.MORPH_OPEN, kernel)
    # save_image(vessels_open, process_results_folder, "6_vessels_open" + "_ch" + str(channel) + ".png")

    # vessels_close = cv2.morphologyEx(inpaint_img, cv2.MORPH_CLOSE, kernel)
    # save_image(vessels_close, process_results_folder, "6_vessels_close" + "_ch" + str(channel) + ".png")


    gsub = gaussian_subtraction(inpaint_img, sigma=5)
    save_image(gsub, process_results_folder, "5_gsub" + "_ch" + str(channel) + ".png")

    # gdiv = gaussian_division(inpaint_img, sigma=5)
    # save_image(gdiv, process_results_folder, "5_gdiv" + "_ch" + str(channel) + ".png")

    # clahe_img = clahe(gsub)
    # clahe2_img = multi_clahe(gsub, 2)
    # clahe3_img = multi_clahe(gsub, 3)

    # save_image(clahe_img, process_results_folder, "5_clahe_img" + "_ch" + str(channel) + ".png")
    # save_image(clahe2_img, process_results_folder, "5_clahe2_img" + "_ch" + str(channel) + ".png")
    # save_image(clahe3_img, process_results_folder, "5_clahe3_img" + "_ch" + str(channel) + ".png")
    
    thresh_img = gsub

    # ostu_thresh = thresh_img > threshold_otsu(thresh_img)
    # save_image(ostu_thresh, process_results_folder, "5_ostu_thresh" + "_ch" + str(channel) + ".png")
    
    thresh_multi_ostu = threshold_multiotsu(thresh_img)
    multi_ostu_r1 = thresh_img > thresh_multi_ostu[0]
    save_image(multi_ostu_r1, process_results_folder, "6_multi_ostu_r1" + "_ch" + str(channel) + ".png")

    # multi_ostu_r2 = thresh_img > thresh_multi_ostu[1]
    # save_image(multi_ostu_r2, process_results_folder, "5_multi_ostu_r2" + "_ch" + str(channel) + ".png")

    inv_img = ~multi_ostu_r1.astype(bool)
    save_image(inv_img, process_results_folder, "7_inv_img" + "_ch" + str(channel) + ".png")

    lab = label(inv_img, connectivity=2)
    keep_vessel = np.zeros_like(inv_img, dtype=bool)
    for r in regionprops(lab):
        if r.area >= 10:     # area == #skeleton pixels
            keep_vessel[lab == r.label] = True

    save_image(keep_vessel, process_results_folder, "8_keep_vessel" + "_ch" + str(channel) + ".png")

    skeleton_img = skeletonize(keep_vessel)
    save_image(skeleton_img, process_results_folder, "9_skeleton_img" + "_ch" + str(channel) + ".png")

    dist = distance_transform_edt(keep_vessel)
    thickness_map = np.zeros_like(dist, dtype=np.float32)
    thickness_map[skeleton_img] = 2.0 * dist[skeleton_img]
    thickness = thickness_map.copy()
    thickness = thickness / thickness.max()
    save_image(thickness, process_results_folder, "9_thickness" + "_ch" + str(channel) + ".png")


    # gblur_thickness = gaussian_blur(thickness, sigma=15)
    # gblur_thickness = gblur_thickness / gblur_thickness.max()
    # save_image(gblur_thickness, process_results_folder, "7_gblur_thickness" + "_ch" + str(channel) + ".png")

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    # dilate1 = cv2.dilate(major_vessels.astype(np.uint8) * 255, kernel, iterations = 1)
    # save_image(dilate1, process_results_folder, "7_dilate1" + "_ch" + str(channel) + ".png")

    
def detect_disc(img):
        
    detect_disc_channel(img, 2)
