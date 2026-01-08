from PIL import Image
import os
from pathlib import Path

from border_processing import create_mask, inpaint
from initial_processing import extract_channel
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
    save_image(per_img, process_results_folder, "5_per_img.png")

    if len(candidates) == 0:
        # print("Fall Back")
        # Percentile Controlled Gamma Enhancement
        per_gamma_img = percentile_controlled_gamma(img)
        pyramid = build_scale_space_DoG_pyramid(per_gamma_img)
        candidates = find_DoG_candidates(pyramid)
        save_image(per_gamma_img, process_results_folder, "5_per_gamma_img.png")

    if len(candidates) == 0:
        # print("Fall Back 2")
        # Percentile Controlled Gamma Enhancement
        clahe_img = clahe(img)
        pyramid = build_scale_space_DoG_pyramid(clahe_img)
        candidates = find_DoG_candidates(pyramid)
        save_image(clahe_img, process_results_folder, "5_clahe_img.png")

    return candidates

def detect_disc_channel(img, channel):

    ch_img = extract_channel(img, channel)

    # Mask to remove dark border
    fov_mask = create_mask(ch_img)
    
    # Fill the border with OpenCV inpaint algorithm
    inpaint_img = inpaint(ch_img, fov_mask)
    
    # Get Disc Candidates
    candidates = get_candidates(inpaint_img)
    # print("Number of Candidates: " + str(len(candidates)))

    # Find best candidate
    best = best_disc_candidate(ch_img, candidates, fov_mask)

    save_image(ch_img, process_results_folder, "2_ch_img" + "_ch" + str(channel) + ".png")
    save_mask_overlay(ch_img, fov_mask, process_results_folder, "3_mask_overlay" + "_ch" + str(channel) + ".png")
    save_image(inpaint_img, process_results_folder, "4_inpaint_img" + "_ch" + str(channel) + ".png")
    save_candidate_overlay(ch_img, candidates, process_results_folder, "6_candidate_overlay" + "_ch" + str(channel) + ".png")
    save_candidate_overlay(ch_img, [best], process_results_folder, "7_best_candidate_overlay" + "_ch" + str(channel) + ".png")

    return best


def detect_disc(img):
        
    best = detect_disc_channel(img, 1)
    print(best)

    # best = detect_disc_channel(img, 2)
    # print(best)

    # best = detect_disc_channel(img, 3)
    # print(best)

    # save_candidate_overlay(red_img, [best], results_folder, Path(test_image).name)

    return best

