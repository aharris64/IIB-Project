import numpy as np

def score_candidate(img, centre, radius, response, 
                    inner_k=1.0, outer_k=1.8, gamma=1.0,
                    w_contrast=1.0, w_brightness=0.7, w_response=0.2):
    """
    Score a blob candidate by combining:
      - local contrast (inner disk vs surrounding annulus),
      - absolute brightness of the candidate,
      - magnitude of the DoG response

    inner_k: scale factor applied to radius to define inner disk radius
    outer_k: scale factor applied to radius to define outer disk radius
    gamma: exponent applied ot brightness term
    w_contrast, w_brightness, w_response: weight of different factors
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
    brightness = mu_in ** gamma

    total_score = w_contrast * contrast + w_brightness * brightness + w_response * abs(response)

    return (total_score, contrast, brightness, response)

def best_disc_candidate(img, candidates, mask):
    """
    Select the best optic disc candidate from a list using a scoring function
    Return tuple of centre and radius of the best candidate
    """

    best = None
    best_score = [-np.inf]

    # if len(candidates) == 1:
    #     return candidates[0]

    for (centre, radius, response) in candidates:
        x, y = centre

        # Check if centre lies outside fov
        if mask[int(y), int(x)] == 0:
            continue

        score = score_candidate(img, centre, radius, response)
        
        if score[0] > best_score[0]:
            best_score = score
            best = (centre, radius, best_score)

    return best