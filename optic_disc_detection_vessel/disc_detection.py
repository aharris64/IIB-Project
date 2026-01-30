from PIL import Image
import os
from pathlib import Path
import cv2
import numpy as np

from border_processing import create_mask, inpaint
from initial_processing import extract_channel
from save_results import save_image, save_mask_overlay

from vessel_enhancement import gaussian_subtraction, otsu_thresholding
from vessel_processing import inverse_bool_img, isolate_major_vessels, vessel_skeleton, vessel_thickness_skeleton

from vessel_density import vessel_density_img
from vessel_cluster import vessel_clusters_and_dist
from vessel_direction import cluster_directions

current_path = Path(__file__).resolve().parent
test_images = os.path.join(current_path, "test_images")
process_results_folder = os.path.join(current_path, "process_results")
results_folder = os.path.join(current_path, "results")
candidate_folder = os.path.join(current_path, "candidates")

def img_processing(img, channel):
    save_image(img, process_results_folder, "1_img.png")

    ch_img = extract_channel(img, channel)
    save_image(ch_img, process_results_folder, "2_ch_img.png")

    # Mask to remove dark border
    fov_mask = create_mask(ch_img)
    save_mask_overlay(ch_img, fov_mask, process_results_folder, "3_mask_overlay.png")

    # Fill the border with OpenCV inpaint algorithm
    inpaint_img = inpaint(ch_img, fov_mask)
    save_image(inpaint_img, process_results_folder, "4_inpaint_img.png")

    return inpaint_img

def vessel_processing(img):

    # Gaussian subtraction to highlight vessels
    gsub = gaussian_subtraction(img, sigma=5)
    save_image(gsub, process_results_folder, "5_gsub.png")
    
    # Otsu thresholding to highlight vessels
    threshold_img = otsu_thresholding(gsub)
    save_image(threshold_img, process_results_folder, "6_threshold_img.png")

    # Inverse image for morphology
    inv_img = inverse_bool_img(threshold_img)
    save_image(inv_img, process_results_folder, "7_inv_img.png")

    # Isolate major vessels
    major_vessels = isolate_major_vessels(inv_img)
    save_image(major_vessels, process_results_folder, "8_major_vessels.png")

    # Skeletonise
    skeleton_img = vessel_skeleton(major_vessels)
    save_image(skeleton_img, process_results_folder, "9_skeleton_img.png")

    # Thickness weighted skeletonise
    thickness_skel = vessel_thickness_skeleton(major_vessels, skeleton_img)
    save_image(thickness_skel, process_results_folder, "9_thickness_skel.png")

    return skeleton_img, thickness_skel

def vessel_density(thickness_skel):

    vessel_density = vessel_density_img(thickness_skel)
    save_image(vessel_density, process_results_folder, "10_vessel_density" + "_ch" + str(channel) + ".png")

def vessel_cluster(skeleton_img):

    cluster_dist, regions_img, cluster_count = vessel_clusters_and_dist(skeleton_img)
    save_image(regions_img, process_results_folder, "11_regions_img" + "_ch" + str(channel) + ".png")
    save_image(cluster_dist, process_results_folder, "11_cluster_dist" + "_ch" + str(channel) + ".png")
    save_image(1.0 - cluster_dist, process_results_folder, "11_inv_cluster_dist" + "_ch" + str(channel) + ".png")

def vessel_cluster_direction(skeleton_img, thickness_skel):

    cluster_directions(skeleton_img, thickness_skel)

def detect_disc(img):

    img = img_processing(img, 2)
        
    skeleton_img, thickness_skel = vessel_processing(img)

    results = vessel_cluster_direction(skeleton_img, thickness_skel)
