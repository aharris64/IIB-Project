"""Single-image entry point wrapping combined_method's full localisation pipeline —
used by both run_disc_localisation_single_image.py and run_disc_localisation_full_dataset.py."""

import sys
from PIL import Image
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from optic_disc_localisation.combined_method.combined_pipeline import optic_disc_localisation
from optic_disc_localisation.image_processing.initial_processing import resize

from optic_disc_localisation.visualisations.save_visualisations import save_vessel_centre_and_blob_candidate, save_centre_overlay

def detect_disc(img_path, target_size=512, save_final=False, save_final_path = False, save_intermediate=False, save_intermediate_path=False):
    """Resize img_path and run the combined localisation pipeline on it.

    Returns (blob_centre, blob_radius, vessel_centre, (score, contrast, response,
    vessel_sign)) — centre/radius/score are None if no blob candidate was found.
    """
    img_name = Path(img_path).name

    with Image.open(img_path) as img:
        img = img.convert("RGB")
        img = resize(img, target_size)
    
    fov_mask, result = optic_disc_localisation(img, save_results=save_intermediate, save_path=save_intermediate_path)

    blob_centre, blob_radius, vessel_centre, (total_score, contrast, response, vessel_sign) = result

    if blob_centre is None:
        if save_final:
            save_centre_overlay(img, vessel_centre, save_final_path, img_name)
        return (None, None, vessel_centre, (None, None, None, None))

    if save_final:
        blob = (blob_centre, blob_radius, total_score)
        save_vessel_centre_and_blob_candidate(img, blob, vessel_centre, save_final_path, img_name)

    return (blob_centre, blob_radius, vessel_centre, (total_score, contrast, response, vessel_sign))
        
