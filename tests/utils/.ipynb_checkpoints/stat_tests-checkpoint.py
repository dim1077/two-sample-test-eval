import numpy as np
from scipy.spatial.distance import cdist
from ot import emd2

def wasserstein_test_perm(x, y, n_permutations=100):
    """
    Multivariate two-sample test via Earth Mover's Distance (Wasserstein).
    x, y: arrays of shape (n_samples, n_features)
    Returns (statistic, p_value).
    """
    cost = cdist(x, y)  
    real_stat = emd2([], [], cost)

    combined = np.vstack([x, y])
    n = x.shape[0]
    stats = []
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        xp, yp = combined[:n], combined[n:]
        stats.append(emd2([], [], cdist(xp, yp)))

    stats = np.array(stats)
    p_value = (np.sum(stats >= real_stat) + 1) / (n_permutations + 1)
    return real_stat, p_value


def _kl_divergence_gaussian(x, y):
    """
    Compute KL( N(mu_x, Sigma_x) || N(mu_y, Sigma_y) )
    for two samples x, y of shape (n, d).
    """
    mu_x = np.mean(x, axis=0)
    mu_y = np.mean(y, axis=0)
    Sigma_x = np.cov(x, rowvar=False)
    Sigma_y = np.cov(y, rowvar=False)
    d = x.shape[1]
    inv_Sigma_y = np.linalg.inv(Sigma_y)
    term1 = np.trace(inv_Sigma_y @ Sigma_x)
    diff = (mu_y - mu_x)[..., None]
    term2 = float(diff.T @ inv_Sigma_y @ diff)
    term3 = np.log(np.linalg.det(Sigma_y) / np.linalg.det(Sigma_x))
    return 0.5 * (term1 + term2 - d + term3)

def kl_test_perm(x, y, n_permutations=100, random_state=None):
    """
    Permutation test using Gaussian‐KL as the statistic.
    Returns (observed_KL, p_value).
    """
    if random_state is not None:
        np.random.seed(random_state)
    x_np = x if isinstance(x, np.ndarray) else x.to_numpy()
    y_np = y if isinstance(y, np.ndarray) else y.to_numpy()
    stat_obs = _kl_divergence_gaussian(x_np, y_np)
    
    pooled = np.vstack([x_np, y_np])
    n = x_np.shape[0]
    perm_stats = []
    for _ in range(n_permutations):
        idx = np.random.permutation(pooled.shape[0])
        x_perm = pooled[idx[:n]]
        y_perm = pooled[idx[n:]]
        perm_stats.append(_kl_divergence_gaussian(x_perm, y_perm))
    perm_stats = np.array(perm_stats)
    
    p_val = np.mean(perm_stats >= stat_obs)
    return stat_obs, p_val