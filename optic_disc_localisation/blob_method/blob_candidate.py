"""Difference-of-Gaussian (DoG) blob detection, and the enhancement-fallback chain
that generates candidates from a vessel-suppressed image."""

import cv2
import numpy as np

from optic_disc_localisation.image_processing.contrast_enhancement import (
    percentage_based_enhancement, percentile_controlled_gamma, clahe
)
from optic_disc_localisation.visualisations.save_visualisations import save_image, save_candidate_overlay


def find_DoG_candidates(img):
    """Run the DoG scale-space pyramid + local-minima detector; return blob candidates."""
    pyramid = build_scale_space_DoG_pyramid(img)

    candidates = find_DoG_minima(pyramid)

    return candidates


def get_candidates(img, save_results=False, save_path=None,
                    percentage_filename="per_img.png",
                    percentile_gamma_filename="per_gamma_img.png",
                    clahe_filename="pclahe_img.png",
                    candidates_filename="candidates.png"):
    """Try percentage-based enhancement, then percentile-controlled gamma, then CLAHE,
    stopping at the first enhancement that yields any DoG blob candidates.

    Shared by blob_method and combined_method (previously duplicated identically in
    each, save for their debug-image filenames — preserved here via the filename
    parameters so callers keep their own existing output naming exactly).
    """
    per_img = percentage_based_enhancement(img)
    candidates = find_DoG_candidates(per_img)

    if save_results:
        save_image(per_img, save_path, percentage_filename)
        if len(candidates) > 0:
            save_candidate_overlay(img, candidates, save_path, candidates_filename)

    if len(candidates) == 0:
        per_gamma_img = percentile_controlled_gamma(img)
        candidates = find_DoG_candidates(per_gamma_img)

        if save_results:
            save_image(per_gamma_img, save_path, percentile_gamma_filename)
            if len(candidates) > 0:
                save_candidate_overlay(img, candidates, save_path, candidates_filename)

    if len(candidates) == 0:
        clahe_img = clahe(img)
        candidates = find_DoG_candidates(clahe_img)

        if save_results:
            save_image(clahe_img, save_path, clahe_filename)
            if len(candidates) > 0:
                save_candidate_overlay(img, candidates, save_path, candidates_filename)

    return candidates

def build_scale_space_DoG_pyramid(img, num_octaves=5, scales_per_octave=3, sigma0=16.0):
    """
    Build a Difference of Gaussian (DoG) scale space pyramid for scale invariant blob detection 
    Produces dark blob on light background as larger blur minus smaller blur will produce negative values at centre

    num_octaves: Maximum number of octaves to generate (each is a factor 2 relative to the previous)
    scales_per_octave: Number of discrete scale intervals per octave,
        the scale step is k = 2^(1 / scales_per_octave)
    sigma0: Base Gaussian standard deviation for the first level

    Returns a scale-space pyramid where each element corresponds to one octave and contains (list of dict)
        'gauss' : Gaussian-blurred images at different scales (list of array)
        'dog' : Difference-of-Gaussians images (gauss[i+1] - gauss[i]) (list of array)
        'sigmas' : Absolute Gaussian sigma values used for each scale (list of float)
    """

    k = 2 ** (1.0 / scales_per_octave) # Scale factor between successive Gaussian levels in an octave

    pyramid = []
    base = img.copy()

    for o in range(num_octaves):
        gaussian_imgs = []
        sigmas = []

        for i in range(scales_per_octave + 3):
            sigma = sigma0 * (k ** i) # Calculate sigma for the current scale
            gaussian_imgs.append(cv2.GaussianBlur(base, (0, 0), sigma))
            sigmas.append(sigma)

        # Calculate Difference of Gaussians (DoG) for each image
        # DoG is an approximation of Laplacian of Gaussian
        DoG = [gaussian_imgs[i+1] - gaussian_imgs[i] for i in range(len(gaussian_imgs)-1)] # Calculae D
        
        pyramid.append({"gauss": gaussian_imgs, "dog": DoG, "sigmas": sigmas})
        
        # Downsampling by a factor of two for next octave
        base = cv2.resize(base, (base.shape[1]//2, base.shape[0]//2), interpolation=cv2.INTER_AREA)

        # Stop if image becomes too small
        if base.shape[0] < 32 or base.shape[1] < 32:
            break

    return pyramid

def map_to_base_resolution(octave, centre, sigma):
    """
    Map a keypoint detected in octave coordinates back to base-resolution space
    """

    scale = 2 ** octave
    centre = (int(round(centre[0] * scale)), int(round(centre[1] * scale)))
    r = np.sqrt(2) * sigma * scale

    return centre, r


def find_DoG_minima(pyramid, threshold=0.001): # Minima for Optic Disc
    """
    Detects dark blob candidates as local minima in DoG scale space pyramid

    Returns a list of candidstes as a list of tuples:
        (x, y, octave, scale_index, sigma, response)
    Coordinates returned are in the octave image coordinate system.
    """
    candidates = []

    for o, octave in enumerate(pyramid):
        DoG = octave["dog"]
        sigmas = octave["sigmas"]

        for i in range(1, len(DoG)-1):
            D = DoG[i]
            prevD = DoG[i-1]
            nextD = DoG[i+1]

            center = D[1:-1, 1:-1] # exclude borders (for 3x3 neighbourhood)
            mask = center <= -threshold # Only keep sufficiently negative responses

            neighbor_slices = []

            for S in (prevD, D, nextD):
                # For each scale plane, take the 8 spatial neighbors (3x3 minus center)
                neighbor_slices.extend([
                    S[0:-2, 0:-2], S[0:-2, 1:-1], S[0:-2, 2:  ],
                    S[1:-1, 0:-2],                 S[1:-1, 2:  ],
                    S[2:  , 0:-2], S[2:  , 1:-1], S[2:  , 2:  ],
                ])

                # Add the (0,0) neighbor only for adjacent scales
                if S is not D:
                    neighbor_slices.append(S[1:-1, 1:-1])

            # Now reduce to a single array: per-pixel min of all neighbors
            neigh_min = np.minimum.reduce(neighbor_slices)

            # The centre must be the minimum of all the neighbours
            candidate = mask & (center <= neigh_min)

            # Extract candidate coordinates
            ys, xs = np.where(candidate)

            sigma = sigmas[i + 1]
            for y0, x0 in zip(ys, xs):
                # Add 1 to undo border cropping
                x = int(x0 + 1)
                y = int(y0 + 1)
                centre = (x, y)
                centre, radius = map_to_base_resolution(o, centre, sigma)
                candidates.append((centre, radius, float(D[y, x])))

    return candidates