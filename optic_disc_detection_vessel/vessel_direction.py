import numpy as np

def weighted_pca_points(points, thickness, thickness_weight=1.0):

    weights = thickness * thickness_weight
    weights_norm = weights / weights.sum()

    mu = (weights_norm[:, None] * points).sum(axis=0)
    Y = points - mu
    C = (Y.T * weights_norm) @ Y  # 2x2 weighted covariance

    evals, evecs = np.linalg.eigh(C)  # ascending
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    direction = evecs[:, order[0]]
    direction = direction / (np.linalg.norm(direction))

    return direction, mu, evals