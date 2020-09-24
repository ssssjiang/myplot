import numpy as np
from myplot_tools import MyException

class GeometryException(MyException):
    pass

def umeyama_alignment(x, y, with_scale=False):
    if x.shape != y.shape:
        raise GeometryException("data matrices must have the same shape.")
    m, n = x.shape

    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)

    # variance
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis])**2)

    # covariance
    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = np.multiply(1.0 / n, outer_sum)

    u, d, v = np.linalg.svd(cov_xy)
    if np.count_nonzero(d > np.finfo(d.dtype).eps) < m - 1:
        raise GeometryException("Degenerate covariance rank, "
                                "Umeyama alignment is not possible")

    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        s[m - 1][n - 1] = -1

    r = u.dot(s).dot(v)

    c = 1 / sigma_x * np.trace(np.diag(d).dot(s)) if with_scale else 1.0
    t = mean_y - np.multiply(c, r.dot(mean_x))

    return r, t, c






