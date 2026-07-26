"""Candidate scoring for combined_method: unlike blob_method's version, this adds a
vessel-convergence-proximity term (score_vessel_blob) as a disambiguating feature and
uses fixed weights fitted by ratings/scripts/combined_candidates/learn_weights.py via
linear regression against manual quality ratings, rather than blob_method's simple
additive weights."""

import numpy as np

def score_vessel_blob(center, radius, point):
    """Proximity score in [~-inf, 1]: 1 at point==center, decreasing linearly to 0 at
    the candidate's radius and negative beyond it."""
    cx, cy = center
    x, y = point

    d = np.hypot(x - cx, y - cy)   # Euclidean distance

    score = 1.0 - d / radius

    return score

def score_candidate(img, centre, radius, response, vessel_centre,
                    inner_k=1.0, outer_k=1.8, w_bias=0.5929768734602114,
                    w_contrast=3.831443 , w_response=-8.454572, w_vessel_sign=1.211922):
    """
    Score a blob candidate by combining:
      - local contrast (inner disk vs surrounding annulus),
      - magnitude of the DoG response,
      - whether the vessel-convergence point lies within the candidate's radius
        (score_vessel_blob's sign, via w_vessel_sign)

    inner_k: scale factor applied to radius to define inner disk radius
    outer_k: scale factor applied to radius to define outer disk radius
    w_bias, w_contrast, w_response, w_vessel_sign: fitted weights (see module docstring)
    """

    h, w = img.shape
    x, y = centre

    r_in  = max(3, int(round(inner_k * radius))) # Enforce minimum of 3 pixels
    r_out = max(r_in + 2, int(round(outer_k * radius))) # Enforce minim of r_in + 2 pixels

    # Compute a square image patch centered on the candidate
    x1 = max(0, x - r_out)
    x2 = min(w, x + r_out + 1)
    y1 = max(0, y - r_out)
    y2 = min(h, y + r_out + 1)

    # Create patch on image 
    patch = img[y1:y2, x1:x2]
    ph, pw = patch.shape
    yy, xx = np.ogrid[0:ph, 0:pw] # Produce coordinates in patch frame
    cxp = x - x1   # center x inside patch
    cyp = y - y1   # center y inside patch

    d2 = (xx - cxp)**2 + (yy - cyp)**2
    inner = d2 <= r_in**2   # Selects pixels d <= r_in
    outer = (d2 > r_in**2) & (d2 <= r_out**2)   # Selects pixels r_in < d <= r_out

    # Check at least one pixel inside inner and outer disc
    if not np.any(inner) or not np.any(outer):
        return -np.inf

    mu_in  = float(np.mean(patch[inner]))
    mu_out = float(np.mean(patch[outer]))

    contrast = mu_in - mu_out

    vessel_score = score_vessel_blob(centre, radius, vessel_centre)
    vessel_sign = (vessel_score > 0).astype(int)

    total_score = w_bias + w_contrast * contrast + w_response * response + w_vessel_sign * vessel_sign

    return (total_score, contrast, response, vessel_sign)

def best_disc_candidate(img, vessel_centre, candidates, mask):
    """
    Select the best optic disc candidate from a list using a scoring function
    Return tuple of centre and radius of the best candidate
    """

    best = (None, None, vessel_centre, (None, None, None, None))
    best_score = [-np.inf]

    h, w = img.shape[:2]

    for (centre, radius, response) in candidates:
        x, y = centre

        # Check if centre lies outside fov
        if mask[int(y), int(x)] == 0:
            continue

        # Check if diameter is greater than image
        if 2 * radius > min(h, w):
            continue

        score = score_candidate(img, centre, radius, response, vessel_centre)
        
        if score[0] > best_score[0]:
            best_score = score
            best = (centre, radius, vessel_centre, best_score)

    return best