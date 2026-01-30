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
from vessel_direction import weighted_pca_points
from vessel_cluster import vessel_clusters_and_dist


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

    # Gaussian subtraction to highlight vessels
    gsub = gaussian_subtraction(inpaint_img, sigma=5)
    save_image(gsub, process_results_folder, "5_gsub" + "_ch" + str(channel) + ".png")
    
    # Otsu thresholding to highlight vessels
    thresold_img = otsu_thresholding(gsub)
    save_image(thresold_img, process_results_folder, "6_multi_ostu_r1" + "_ch" + str(channel) + ".png")

    # Inverse image for morphology
    inv_img = inverse_bool_img(thresold_img)
    save_image(inv_img, process_results_folder, "7_inv_img" + "_ch" + str(channel) + ".png")

    # Isolate major vessels
    major_vessels = isolate_major_vessels(inv_img)
    save_image(major_vessels, process_results_folder, "8_major_vessels" + "_ch" + str(channel) + ".png")

    # Skeletonise
    skeleton_img = vessel_skeleton(major_vessels)
    save_image(skeleton_img, process_results_folder, "9_skeleton_img" + "_ch" + str(channel) + ".png")

    # Thickness weighted skeletonise
    thickness_skel = vessel_thickness_skeleton(major_vessels, skeleton_img)
    save_image(thickness_skel, process_results_folder, "9_thickness_skel" + "_ch" + str(channel) + ".png")

    # -----------------------------------------
    #           Vessel Based Methods
    # -----------------------------------------

    # ---------- Vessel Density -------------
    vessel_density = vessel_density_img(thickness_skel)
    save_image(vessel_density, process_results_folder, "10_vessel_density" + "_ch" + str(channel) + ".png")


    # ---------- Cluster Distance -------------
   
    cluster_dist, regions_img, cluster_count = vessel_clusters_and_dist(skeleton_img)
    save_image(regions_img, process_results_folder, "11_regions_img" + "_ch" + str(channel) + ".png")
    save_image(cluster_dist, process_results_folder, "11_cluster_dist" + "_ch" + str(channel) + ".png")
    save_image(1.0 - cluster_dist, process_results_folder, "11_inv_cluster_dist" + "_ch" + str(channel) + ".png")

    # ------- Vessel Direction -------

    




def detect_disc(img):
        
    detect_disc_channel(img, 2)
