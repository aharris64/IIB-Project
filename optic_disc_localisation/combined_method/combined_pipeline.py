from optic_disc_localisation.image_processing.fov_processing import create_mask, inpaint
from optic_disc_localisation.image_processing.initial_processing import extract_channel
from optic_disc_localisation.image_processing.contrast_enhacement import  percentage_based_enhancement, percentile_controlled_gamma, clahe
from optic_disc_localisation.image_processing.gaussian_processing import  gaussian_subtraction, gaussian_blur

from optic_disc_localisation.blob_method.blob_candidate import find_DoG_candidates
from optic_disc_localisation.blob_method.candidate_evaluation import best_disc_candidate

from optic_disc_localisation.image_processing.vessel_processing import inverse_bool_img, otsu_thresholding, inverse_bool_img, isolate_major_vessels, vessel_skeleton, vessel_thickness_skeleton, vessel_inpaint
from optic_disc_localisation.vessel_method.vessel_directions import pca_on_grid_boxes
from optic_disc_localisation.vessel_method.vessel_convergence import generate_vessel_rays, find_convergence_point, blur_rays

from optic_disc_localisation.visualisations.save_visualisations import save_image, save_mask_overlay, save_candidate_overlay, save_grid_pca, save_centre_overlay, save_vessel_centre_and_blob_candidate

def image_processing(img, channel, save_results=False, save_path=None):

    # Red channel
    ch_img = extract_channel(img, channel)

    # Mask to remove dark border
    fov_mask = create_mask(ch_img)
    
    # Fill the border with OpenCV inpaint algorithm
    inpaint_img = inpaint(ch_img, fov_mask)

    if save_results:
        save_image(ch_img, save_path, f"1_ch{channel}_img.png")
        save_mask_overlay(ch_img, fov_mask, save_path, f"2_ch{channel}_mask_overlay.png")
        save_image(inpaint_img, save_path, f"3_ch{channel}_inpaint_img.png")

    return fov_mask, inpaint_img

def vessel_extraction(rgb_img, save_results=False, save_path=None):

    # Green channel
    fov_mask, inpaint_img = image_processing(rgb_img, 2, save_results=save_results, save_path=save_path)

    # Gaussian subtraction to highlight vessels
    gsub = gaussian_subtraction(inpaint_img, 5)
    
    # Otsu thresholding to highlight vessels
    threshold_img = otsu_thresholding(gsub)

    # Inverse image for morphology
    inv_img = inverse_bool_img(threshold_img)

    if save_results:
        save_image(inpaint_img, save_path, "4_inpaint_img.png")
        save_image(gsub, save_path, "5_gsub.png")
        save_image(threshold_img, save_path, "6_threshold_img.png")
        save_image(inv_img, save_path, "7_inv_img.png")

    return inv_img

def vessel_processing(vessel_mask_img, save_results=False, save_path=None):
    
    # Isolate major vessels
    major_vessels = isolate_major_vessels(vessel_mask_img)
    
    # Skeletonise
    skeleton_img = vessel_skeleton(major_vessels)
    
    # Thickness weighted skeletonise
    thickness_skel = vessel_thickness_skeleton(major_vessels, skeleton_img)

    if save_results:
        save_image(major_vessels, save_path, "8_major_vessels.png")
        save_image(skeleton_img, save_path, "9_skeleton_img.png")
        save_image(thickness_skel, save_path, "10_thickness_skel.png")

    return skeleton_img, thickness_skel

def vessel_convergence(skeleton_img, thickness_skel, save_results=False, save_path=None):

    # Perform PCA on grid
    grid_results = pca_on_grid_boxes(skeleton_img, thickness_skel)
    
    # Extend and attenuate rays
    rays = generate_vessel_rays(skeleton_img, grid_results)

    # Blur rays
    blurred = blur_rays(rays)

    # Find peak of blurred rays
    p_xy = find_convergence_point(blurred)

    if save_results:
        save_grid_pca(skeleton_img, grid_results, save_path, "11_dir_grid_img.png")
        save_image(rays, save_path, "12_rays.png")
        save_image(blurred, save_path, "13_blurred.png")

    return p_xy

def vessel_suppression(vessel_mask_img, red_img, save_results=False, save_path=None):

    # Vessel inpaint
    vessel_removed_img = vessel_inpaint(red_img, vessel_mask_img, save_results=save_results, save_path=save_path)

    if save_results:
        save_image(vessel_removed_img, save_path, "14_vessel_removed_img.png")
    
    return vessel_removed_img

def get_candidates(img, save_results=False, save_path=None):
    
    # Percentage Based Enhancement
    per_img = percentage_based_enhancement(img)
    candidates = find_DoG_candidates(per_img)

    if save_results:
        save_image(per_img, save_path, "15_per_img.png")
        if len(candidates) > 0:
            save_candidate_overlay(img, candidates, save_path, "16_candidates.png")

    if len(candidates) == 0:
        # Percentile Controlled Gamma Enhancement
        per_gamma_img = percentile_controlled_gamma(img)
        candidates = find_DoG_candidates(per_gamma_img)

        if save_results:
            save_image(per_gamma_img, save_path, "15_per_gamma_img.png")
            if len(candidates) > 0:
                save_candidate_overlay(img, candidates, save_path, "16_candidates.png")

    if len(candidates) == 0:
        # Percentile Controlled Gamma Enhancement
        clahe_img = clahe(img)
        candidates = find_DoG_candidates(clahe_img)

        if save_results:
            save_image(clahe_img, save_path, "15_pclahe_img.png")
            if len(candidates) > 0:
                save_candidate_overlay(img, candidates, save_path, "16_candidates.png")

    return candidates

def blob_disc_detection(img, save_results=False, save_path=None):

    fov_mask, processed_img = image_processing(img, save_results=save_results, save_path=save_path)

    # Vessel Extraction
    vessel_mask = vessel_extraction(img, save_results=save_results, save_path=save_path)
    
    # Inpaint vessels 
    vessel_suppressed_img = vessel_suppression(vessel_mask, processed_img, save_results=save_results, save_path=save_path)

    # Get Disc Candidates
    candidates = get_candidates(vessel_suppressed_img, save_results=save_results, save_path=save_path)

    # Vessel processing
    skeleton_img, thickness_skel = vessel_processing(vessel_mask, save_results=save_results, save_path=save_path)

    # Find centre
    vessel_centre = vessel_convergence(skeleton_img, thickness_skel, save_results=save_results, save_path=save_path)

    # Find best candidate
    best = best_disc_candidate(vessel_suppressed_img, vessel_centre, candidates, fov_mask)

    if save_results:
        save_centre_overlay(img, vessel_centre, save_path, "17_vessel_centre.png")
        save_candidate_overlay(img, [best], save_path, "17_best_candidate.png")
        save_vessel_centre_and_blob_candidate(img, [best], vessel_centre, save_path, "18_vessel_and_best.png")

    return best