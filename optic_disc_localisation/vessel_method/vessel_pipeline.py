from optic_disc_localisation.image_processing.fov_processing import create_mask, inpaint
from optic_disc_localisation.image_processing.initial_processing import extract_channel
from optic_disc_localisation.image_processing.gaussian_processing import gaussian_subtraction
from optic_disc_localisation.image_processing.vessel_processing import inverse_bool_img, otsu_thresholding, inverse_bool_img, isolate_major_vessels, vessel_skeleton, vessel_thickness_skeleton

from optic_disc_localisation.vessel_method.vessel_directions import pca_on_grid_boxes
from optic_disc_localisation.vessel_method.vessel_convergence import generate_vessel_rays, find_convergence_point, blur_rays

from optic_disc_localisation.visualisations.save_visualisations import save_image, save_mask_overlay, save_grid_pca, save_centre_overlay

def img_processing(img, save_results=False, save_path=None):
    
    # Green Channel
    green_img = extract_channel(img, 2)
    
    # Mask to remove dark border
    fov_mask = create_mask(green_img)
    
    # Fill the border with OpenCV inpaint algorithm
    inpaint_img = inpaint(green_img, fov_mask)

    if save_results:
        save_image(green_img, save_path, "v_1_green_img.png")
        save_mask_overlay(green_img, fov_mask, save_path, "v_2_mask_overlay.png")
        save_image(inpaint_img, save_path, "v_3_inpaint_img.png")

    return inpaint_img

def vessel_processing(img, save_results=False, save_path=None):

    # Gaussian subtraction to highlight vessels
    gsub = gaussian_subtraction(img, 5)
    
    # Otsu thresholding to highlight vessels
    threshold_img = otsu_thresholding(gsub)

    # Inverse image for morphology
    inv_img = inverse_bool_img(threshold_img)
    
    # Isolate major vessels
    major_vessels = isolate_major_vessels(inv_img)
    
    # Skeletonise
    skeleton_img = vessel_skeleton(major_vessels)
    
    # Thickness weighted skeletonise
    thickness_skel = vessel_thickness_skeleton(major_vessels, skeleton_img)

    if save_results:
        save_image(gsub, save_path, "v_4_gsub.png")
        save_image(threshold_img, save_path, "v_5_threshold_img.png")
        save_image(inv_img, save_path, "v_6_inv_img.png")
        save_image(major_vessels, save_path, "v_7_major_vessels.png")
        save_image(skeleton_img, save_path, "v_8_skeleton_img.png")
        save_image(thickness_skel, save_path, "v_9_thickness_skel.png")

    return skeleton_img, thickness_skel

def optic_disc_centre(skeleton_img, thickness_skel, save_results=False, save_path=None):

    # Perform PCA on grid
    grid_results = pca_on_grid_boxes(skeleton_img, thickness_skel)
    
    # Extend and attenuate rays
    rays = generate_vessel_rays(skeleton_img, grid_results)

    # Blur rays
    blurred = blur_rays(rays)

    # Find peak of blurred rays
    p_xy = find_convergence_point(blurred)

    if save_results:
        save_grid_pca(skeleton_img, grid_results, save_path, "v_10_dir_grid_img.png")
        save_image(rays, save_path, "v_11_rays.png")
        save_image(blurred, save_path, "v_12_blurred.png")

    return p_xy

def vessel_disc_detection(img, save_results=False, save_path=None):

    # Image processing - green channel
    processed_img = img_processing(img, save_results=save_results, save_path=save_path)
    
    # Vessel processing
    skeleton_img, thickness_skel = vessel_processing(processed_img, save_results=save_results, save_path=save_path)

    # Find centre
    centre = optic_disc_centre(skeleton_img, thickness_skel, save_results=save_results, save_path=save_path)

    if save_results:
        save_centre_overlay(img, centre, save_path, "v_13_vessel_centre.png")

    return centre
