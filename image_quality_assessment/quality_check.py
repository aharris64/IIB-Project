"""Run-time image quality gate for a captured fundus photo: localises the optic disc
then checks focus, disc-detection confidence, and whether the disc lies within the
image/field-of-view bounds, returning a retake message on the first failed check."""

import cv2
from PIL import Image
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parents[1]))

from optic_disc_localisation.combined_method.combined_pipeline import optic_disc_localisation
from optic_disc_localisation.image_processing.initial_processing import resize

FOCUS_THRESHOLD = 0.0
ODL_THRESHOLD = 3.04
# Sample image not included in the repo (optic_disc_localisation/input_images is gitignored) — replace with your own.
image_path = str(Path(__file__).resolve().parents[1] / "optic_disc_localisation" / "input_images" / "normal_0009_EDD.jpg")

def _cardinal_points(centre, r):
    cx, cy = centre
    return [(cx + r, cy), (cx - r, cy), (cx, cy + r), (cx, cy - r)]


def check_in_focus(img, threshold=0):
    """Check the image is in focus using the variance of the Laplacian.

    Returns (True, None) if in focus, or (False, "Please retake: image out of focus").
    """
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    grey = cv2.normalize(grey, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    if cv2.Laplacian(grey, cv2.CV_64F).var() >= threshold:
        return True, None

    return False, "Please retake: image out of focus"


def check_disc_in_bounds(best, img_shape):
    """Check the four cardinal points of the disc lie within the image boundary.

    Returns (True, None) if in bounds, or (False, "Please retake: optic disc not centred").
    """
    centre, r, _, _ = best
    h, w = img_shape

    for px, py in _cardinal_points(centre, r):
        if not (0 <= px < w and 0 <= py < h):
            return False, "Please retake: optic disc not centred"

    return True, None


def check_disc_in_mask(best, fov_mask):
    """Check the four cardinal points of the disc lie within the FOV mask.

    Returns (True, None) if valid, or (False, "Please retake: optic disc not centred").
    """
    centre, r, _, _ = best

    for px, py in _cardinal_points(centre, r):
        xi, yi = int(round(px)), int(round(py))
        if fov_mask[yi, xi] == 0:
            return False, "Please retake: optic disc not centred"

    return True, None


def check_disc_score(best, threshold):
    """Check the disc score meets the minimum threshold.
 
    Returns (True, None) if valid, or (False, "Please retake: optic disc not localised").
    """
    _, _, _, (total_score, _, _, _) = best
 
    if total_score >= threshold:
        return True, None
 
    return False, "Please retake: optic disc not localised"


def image_quality_assessment(img):
    """Run all quality checks in order: focus, score, bounds, mask.

    Stops and returns the first failure message.
    Returns (True, None) if all checks pass.
    """

    fov_mask, best = optic_disc_localisation(img)
    print(best)

    for check, args in [
        (check_in_focus,       (img, FOCUS_THRESHOLD)),
        (check_disc_score,     (best, ODL_THRESHOLD)),
        (check_disc_in_bounds, (best, img.shape[:2])),
        (check_disc_in_mask,   (best, fov_mask)),
    ]:
        passed, msg = check(*args)
        if not passed:
            return False, msg

    return True, None


def detect_disc(img_path, target_size=512):
    """Load img_path, resize it, and run the image quality gate; prints the result."""

    with Image.open(img_path) as img:
        img = img.convert("RGB")
        img = resize(img, target_size)
    
    result, msg = image_quality_assessment(img)

    if result:
        print("Image Passed!")
    else:
        print(msg)
        
detect_disc(image_path)