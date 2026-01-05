from PIL import Image
import os
from pathlib import Path

from border_processing import create_mask, inpaint
from initial_processing import resize, extract_red_channel
from disc_enhancement import  percentage_based_enhancement, percentile_controlled_gamma, clahe
from difference_of_gaussians import build_scale_space_DoG_pyramid, find_DoG_candidates
from scoring import best_disc_candidate
from save_results import save_image, save_mask_overlay, save_candidate_overlay

current_path = Path(__file__).resolve().parent
test_images = os.path.join(current_path, "test_images")
process_results_folder = os.path.join(current_path, "process_results")
results_folder = os.path.join(current_path, "results")
pyramid_folder = os.path.join(current_path, "pyramid")
candidate_folder = os.path.join(current_path, "candidates")

def get_candidates(img):
    
    # Percentage Based Enhancement
    per_img = percentage_based_enhancement(img)
    pyramid = build_scale_space_DoG_pyramid(per_img)
    candidates = find_DoG_candidates(pyramid)

    if len(candidates) == 0:
        # print("Fall Back")
        # Percentile Controlled Gamma Enhancement
        per_gamma_img = percentile_controlled_gamma(img)
        pyramid = build_scale_space_DoG_pyramid(per_gamma_img)
        candidates = find_DoG_candidates(pyramid)

    if len(candidates) == 0:
        # print("Fall Back 2")
        # Percentile Controlled Gamma Enhancement
        clahe_img = clahe(img)
        pyramid = build_scale_space_DoG_pyramid(clahe_img)
        candidates = find_DoG_candidates(pyramid)

    return candidates


def detect_disc(img):
        
    # Extract red channel 
    red_img = extract_red_channel(img) # (Normalised [0, 1] float32 array)
    
    # Mask to remove dark border
    fov_mask = create_mask(red_img)
    
    # Fill the border with OpenCV inpaint algorithm
    inpaint_img = inpaint(red_img, fov_mask)
    
    # Get Disc Candidates
    candidates = get_candidates(inpaint_img)
    # print("Number of Candidates: " + str(len(candidates)))

    # Find best candidate
    best = best_disc_candidate(red_img, candidates, fov_mask)

    # save_image(resize_img, process_results_folder, "1_resized_rgb.png")
    # save_image(red_img, process_results_folder, "2_red_img.png")
    # save_mask_overlay(red_img, fov_mask, process_results_folder, "3_mask_overlay.png")
    # save_image(inpaint_img, process_results_folder, "4_inpaint_img.png")
    # save_image(per_img, process_results_folder, "5_per_img.png")
    # save_image(per_gamma_img, process_results_folder, "5_per_gamma_img.png")
    # save_candidate_overlay(red_img, candidates, process_results_folder, "6_candidate_overlay.png")
    # save_candidate_overlay(red_img, [best], results_folder, "7_best_candidate_overlay.png")

    # save_candidate_overlay(red_img, [best], results_folder, Path(test_image).name)

    return best

