"""util.py"""

def get_penalty(accuracy, reward=1):
    """Return penalty needed for this accuracy to have expected reward 0.

    >>> round(get_penalty(0.9), 10)
    -9.0
    >>> round(get_penalty(0.75), 10)
    -3.0
    >>> round(get_penalty(0.5), 10)
    -1.0

    """
    return accuracy * reward / (accuracy - 1)

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory) 

def truncnorm_sample(lower, upper, mu, std, size=1):
    """Sample from a truncated normal distribution.

    More intuitive version of scipy truncnorm function.

    Args:
        lower:  Lower bound.
        uppper: Upper bound.
        mu:     Mean.
        std:    Standard deviation.
        size:   Number of samples.

    Returns: Numpy array.

    """
    import numpy as np
    import scipy.stats as ss
    if std == 0:
        return np.array([mu for _ in xrange(size)])
    else:
        return ss.truncnorm.rvs((lower - mu) / std, (upper - mu) / std,
                                loc=mu, scale=std, size=size)

