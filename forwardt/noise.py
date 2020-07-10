import numpy as np
from numpy.testing import assert_array_almost_equal


def unbiased_edge(x, y, p_minus, p_plus):
    z = (y - (p_minus - p_plus)) * x
    return z / (1 - p_minus - p_plus)


def unbiased_mean_op(X, y, p_minus, p_plus):
    return np.array([unbiased_edge(X[i, :], y[i], p_minus, p_plus)
                    for i in np.arange(X.shape[0])]).mean(axis=0)


def build_uniform_P(size, noise):
    """ The noise matrix flips any class to any other with probability
    noise / (#class - 1).
    """

    assert(noise >= 0.) and (noise <= 1.)

    P = noise / (size - 1) * np.ones((size, size))
    np.fill_diagonal(P, (1 - noise) * np.ones(size))

    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P



def row_normalize_P(P, copy=True):

    if copy:
        P_norm = P.copy()
    else:
        P_norm = P

    D = np.sum(P, axis=1)
    for i in np.arange(P_norm.shape[0]):
        P_norm[i, :] /= D[i]
    return P_norm
