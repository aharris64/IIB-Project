"""Blob-detection-only optic disc localisation pipeline (no vessel convergence) —
contrast with combined_method/combined_pipeline.py, which fuses this with
vessel_method's vessel-convergence pipeline for a better-disambiguated result."""

from optic_disc_localisation.image_processing.initial_processing import extract_channel_masked
from optic_disc_localisation.image_processing.gaussian_processing import gaussian_subtraction
from optic_disc_localisation.image_processing.vessel_processing import otsu_thresholding, inverse_bool_img, vessel_inpaint

from optic_disc_localisation.blob_method.blob_candidate import get_candidates
from optic_disc_localisation.blob_method.candidate_evaluation import best_disc_candidate

from optic_disc_localisation.visualisations.save_visualisations import save_image, save_mask_overlay, save_candidate_overlay


def vessel_suppression(rgb_img, red_img, save_results=False, save_path=None):
    """Suppress vessels in red_img using a vessel mask derived from the green channel."""

    # Green channel
    green_img, fov_mask, inpaint_img = extract_channel_masked(rgb_img, 2)

    # Gaussian subtraction to highlight vessels
    gsub = gaussian_subtraction(inpaint_img, 5)
    
    # Otsu thresholding to highlight vessels
    threshold_img = otsu_thresholding(gsub)

    # Inverse image for morphology
    inv_img = inverse_bool_img(threshold_img)

    # Vessel inpaint
    vessel_removed_img = vessel_inpaint(red_img, inv_img)

    if save_results:
        save_image(green_img, save_path, "b_4_green_img.png")
        save_mask_overlay(green_img, fov_mask, save_path, "b_5_mask_overlay.png")
        save_image(inpaint_img, save_path, "b_6_inpaint_img.png")
        save_image(gsub, save_path, "b_7_gsub.png")
        save_image(threshold_img, save_path, "b_8_threshold_img.png")
        save_image(inv_img, save_path, "b_9_inv_img.png")
        save_image(vessel_removed_img, save_path, "b_10_vessel_removed_img.png")
    
    return vessel_removed_img

def image_processing(img, save_results=False, save_path=None):
    """Extract, mask, and inpaint the red channel (used as the vessel-suppression base)."""

    # Red channel
    red_img, fov_mask, inpaint_img = extract_channel_masked(img, 1)

    if save_results:
        save_image(red_img, save_path, "b_1_red_img.png")
        save_mask_overlay(red_img, fov_mask, save_path, "b_2_mask_overlay.png")
        save_image(inpaint_img, save_path, "b_3_inpaint_img.png")

    return fov_mask, inpaint_img


def blob_disc_detection(img, save_results=False, save_path=None):
    """Full blob-only pipeline: red-channel prep -> vessel suppression -> DoG blob
    candidates -> best-scoring candidate. Returns (centre, radius, score) or None."""

    fov_mask, processed_img = image_processing(img, save_results=save_results, save_path=save_path)

    # Inpaint vessels
    inpaint_vessles_img = vessel_suppression(img, processed_img, save_results=save_results, save_path=save_path)

    # Get Disc Candidates
    candidates = get_candidates(
        inpaint_vessles_img, save_results=save_results, save_path=save_path,
        percentage_filename="b_11_per_img.png",
        percentile_gamma_filename="b_11_per_gamma_img.png",
        clahe_filename="b_11_pclahe_img.png",
        candidates_filename="b_12_candidates.png",
    )

    # Find best candidate
    best = best_disc_candidate(inpaint_vessles_img, candidates, fov_mask)

    if save_results:
        save_candidate_overlay(img, [best], save_path, "b_13_best_candidate.png")

    return best
