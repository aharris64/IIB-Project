from PIL import Image
import os
from pathlib import Path
import cv2
import numpy as np

from border_processing import create_mask, inpaint
from initial_processing import extract_channel
from save_results import save_image, save_mask_overlay
from render_vessel_lines import render_grid_directions, render_vessel_rays

from vessel_enhancement import gaussian_subtraction, otsu_thresholding
from vessel_processing import inverse_bool_img, isolate_major_vessels, vessel_skeleton, vessel_thickness_skeleton

from grid_pca import pca_on_grid_boxes
from vessel_convergence import generate_vessel_rays, blur_vessel_rays, find_convergence_point

current_path = Path(__file__).resolve().parent
test_images = os.path.join(current_path, "test_images")
process_results_folder = os.path.join(current_path, "process_results")
image_results_folder = os.path.join(current_path, "image_results")

def img_processing(img, channel):
    
    ch_img = extract_channel(img, channel)
    
    # Mask to remove dark border
    fov_mask = create_mask(ch_img)
    
    # Fill the border with OpenCV inpaint algorithm
    inpaint_img = inpaint(ch_img, fov_mask)

    save_image(img, process_results_folder, "1_img.png")
    save_image(ch_img, process_results_folder, "2_ch_img.png")
    save_mask_overlay(ch_img, fov_mask, process_results_folder, "3_mask_overlay.png")
    save_image(inpaint_img, process_results_folder, "4_inpaint_img.png")

    return inpaint_img

def vessel_processing(img):

    # Gaussian subtraction to highlight vessels
    gsub = gaussian_subtraction(img, sigma=5)
    
    # Otsu thresholding to highlight vessels
    threshold_img = otsu_thresholding(gsub)

    # Inverse image for morphology
    inv_img = inverse_bool_img(threshold_img)
    
    # Isolate major vessels
    major_vessels = isolate_major_vessels(inv_img, min_area=10)
    
    # Skeletonise
    skeleton_img = vessel_skeleton(major_vessels)
    
    # Thickness weighted skeletonise
    thickness_skel = vessel_thickness_skeleton(major_vessels, skeleton_img)

    save_image(gsub, process_results_folder, "5_gsub.png")
    save_image(threshold_img, process_results_folder, "6_threshold_img.png")
    save_image(inv_img, process_results_folder, "7_inv_img.png")
    save_image(major_vessels, process_results_folder, "8_major_vessels.png")
    save_image(skeleton_img, process_results_folder, "9_skeleton_img.png")
    save_image(thickness_skel, process_results_folder, "9_thickness_skel.png")

    return skeleton_img, thickness_skel

def optic_disc_centre(skeleton_img, thickness_skel):

    grid_results = pca_on_grid_boxes(skeleton_img, thickness_skel, box_size=10, min_points=5, weight_power=1.0)
    dir_grid_img = render_grid_directions(skeleton_img.shape, grid_results, l=5)
    
    eps = 1e-15

    acc = generate_vessel_rays(skeleton_img.shape[:2], grid_results, sigma=50, eps=eps, use_weights=False)
    acc_out = acc
    acc_out = acc_out / (acc_out.max() + eps)
    

    blurred = blur_vessel_rays(acc, blur_sigma=20)
    p_xy, peak = find_convergence_point(blurred)
    blurred_out = blurred
    blurred_out = blurred_out / (blurred_out.max() + eps)

    save_image(dir_grid_img, process_results_folder, "10_dir_grid_img.png")
    save_image(acc_out, process_results_folder, "11_acc_out.png")
    save_image(blurred_out, process_results_folder, "12_blurred_out.png")

    print("Estimated centre:", p_xy)
    return p_xy

def detect_disc(img, name):

    processed_img = img_processing(img, 2)
    
    skeleton_img, thickness_skel = vessel_processing(processed_img)

    centre = optic_disc_centre(skeleton_img, thickness_skel)

    img_centre = img.copy()
    H, W, _ = img_centre.shape

    x, y = float(centre[0]), float(centre[1])
    r0, c0 = int(round(y)), int(round(x))

    radius = 3
    color=(0, 255,0)

    rr = np.arange(r0 - radius, r0 + radius + 1)
    cc = np.arange(c0 - radius, c0 + radius + 1)
    rr = rr[(rr >= 0) & (rr < H)]
    cc = cc[(cc >= 0) & (cc < W)]

    for r in rr:
        for c in cc:
            if (r - r0)**2 + (c - c0)**2 <= radius**2:
                img_centre[r, c, 0] = color[0]
                img_centre[r, c, 1] = color[1]
                img_centre[r, c, 2] = color[2]

    save_image(img_centre, process_results_folder, "13_img_centre.png")
    # save_image(img_centre, image_results_folder, name)

