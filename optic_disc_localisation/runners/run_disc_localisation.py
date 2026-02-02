import numpy as np
from PIL import Image
from pathlib import Path

from optic_disc_localisation.blob_method.blob_pipeline import blob_disc_detection
from optic_disc_localisation.vessel_method.vessel_pipeline import vessel_disc_detection

from optic_disc_localisation.image_processing.initial_processing import resize

from optic_disc_localisation.visualisations.save_visualisations import save_vessel_centre_and_blob_candidate, save_centre_overlay

def score_vessel_blob(center, radius, point):
    cx, cy = center
    x, y = point

    d = np.hypot(x - cx, y - cy)   # Euclidean distance

    score = 1.0 - d / radius

    return score

def detect_disc(img_path, target_size=512, save_final=False, save_final_path = False, save_intermediate=False, save_intermediate_path=False):
    
    img_name = Path(img_path).name

    with Image.open(img_path) as img:
        img = img.convert("RGB")
        img = resize(img, target_size)
    
    blob_centre, blob_radius, blob_score = blob_disc_detection(img, save_results=save_intermediate, save_path=save_intermediate_path)
    vessel_centre = vessel_disc_detection(img, save_results=save_intermediate, save_path=save_intermediate_path)

    if blob_centre is None or blob_radius is None:
        if save_final:
            save_centre_overlay(img, vessel_centre, save_final_path, img_name)
        return ("no_blob", None, (None, None, None), vessel_centre)

    score = score_vessel_blob(blob_centre, blob_radius, vessel_centre)

    if save_final:
        blob = (blob_centre, blob_radius, blob_score)
        save_vessel_centre_and_blob_candidate(img, blob, vessel_centre, save_final_path, img_name)

    return ("ok", float(score) if score is not None else None,
            (blob_centre, blob_radius, blob_score), vessel_centre)
        
