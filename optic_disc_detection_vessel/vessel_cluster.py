from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt
import numpy as np

def create_vessel_clusters(img):

    # TODO: Check binary image

    clusters = label(img, connectivity=2)

    print("clusters:", len(clusters))

    return

def compute_cluster_distance(img):

    distance = np.zeros_like(img, dtype=np.float32)

    clusters = create_vessel_clusters(img)

    return