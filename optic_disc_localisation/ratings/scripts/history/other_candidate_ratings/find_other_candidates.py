"""Detection logic for the "other candidates" rating set: for each image, finds every
valid blob candidate *except* the single best one (used to build negative/near-miss
examples for learn_weights.py's rating-vs-feature regression). Pure detection module —
no CSV/JSON I/O of its own; used by run_candidate_generation.py."""

from optic_disc_localisation.blob_method.blob_pipeline import image_processing, vessel_suppression, get_candidates
from optic_disc_localisation.visualisations.save_visualisations import save_vessel_centre_and_blob_candidate
from optic_disc_localisation.blob_method.candidate_evaluation import score_candidate
from optic_disc_localisation.image_processing.initial_processing import resize
from optic_disc_localisation.vessel_method.vessel_pipeline import vessel_disc_detection

import numpy as np
from pathlib import Path
from PIL import Image

def score_vessel_blob(center, radius, point):
    """Proximity score: 1 at point==center, decreasing to 0 at radius, negative beyond."""
    cx, cy = center
    x, y = point

    d = np.hypot(x - cx, y - cy)   # Euclidean distance

    score = 1.0 - d / radius

    return score

def detect_disc(img_path, save_candidate_path, target_size=512):
    """Resize img_path, find every non-best blob candidate, save a per-candidate
    overlay to save_candidate_path, and return a list of
    (vessel_proximity_score, (centre, radius, blob_score), vessel_centre)."""

    img_stem = Path(img_path).stem

    with Image.open(img_path) as img:
        img = img.convert("RGB")
        img = resize(img, target_size)
    
    other_candidates = blob_disc_detection(img)
    vessel_centre = vessel_disc_detection(img)

    candidate_list = []

    if len(other_candidates) > 0:
        for k, blob in enumerate(other_candidates, start=1):
            out_name = f"{img_stem}_cand{k:02d}.png"
            save_vessel_centre_and_blob_candidate(img, blob, vessel_centre, save_candidate_path, out_name)
            blob_centre, blob_radius, blob_score = blob
            score = score_vessel_blob(blob_centre, blob_radius, vessel_centre)

            candidate_list.append((float(score), (blob_centre, blob_radius, blob_score), vessel_centre))

    # blob_score = (total_score, contrast, brightness, response)

    return candidate_list

def blob_disc_detection(img):
    """Blob-detect img and return every valid candidate except the best-scoring one."""

    fov_mask, processed_img = image_processing(img)
    
    # Inpaint vessels 
    inpaint_vessles_img = vessel_suppression(img, processed_img)

    # Get Disc Candidates
    candidates = get_candidates(inpaint_vessles_img)

    # Find all candidates apart from best
    others = other_disc_candidates(inpaint_vessles_img, candidates, fov_mask)

    return others

def other_disc_candidates(img, candidates, mask):
    """
    Return all valid optic disc candidates except the best one,
    in the format: (centre, radius, score)
    """

    scored = []

    # Score all valid candidates
    for (centre, radius, response) in candidates:
        x, y = centre

        # Check if centre lies outside fov
        if mask[int(y), int(x)] == 0:
            continue

        score = score_candidate(img, centre, radius, response)

        scored.append((centre, radius, score))

    # If no valid candidates
    if len(scored) == 0:
        return []

    # Find best
    best = max(scored, key=lambda c: c[2][0])

    # Remove best
    others = [
        c for c in scored
        if not (c[0] == best[0] and c[1] == best[1])
    ]

    return others