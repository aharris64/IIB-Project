import cv2
import numpy as np

def vessel_density_img(vessel_skel, sigma = 30):
    vessel_density = cv2.GaussianBlur(vessel_skel.astype(np.float32), (0,0), sigma)
    vessel_density = vessel_density / (vessel_density.max() + 1e-8)

    return vessel_density