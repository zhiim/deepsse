import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import directed_hausdorff


def hausdorff_distance(a, b):
    """Calculate Hausdorff distance between two sets of points.

    Args:
        a (np.ndarray): Set of points
        b (np.ndarray): Set of points
    """
    a = a.reshape(-1, 1)
    b = b.reshape(-1, 1)
    return max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0])


def ospa_distance(set_x, set_y, p=2, c=5):
    if set_x.ndim == 0:
        set_x = set_x[np.newaxis]
    if set_y.ndim == 0:
        set_y = set_y[np.newaxis]
    m = set_x.size
    n = set_y.size
    if m == 0 and n == 0:
        return 0
    elif m == 0 or n == 0:
        return (c**p) ** (1 / p)

    if m > n:
        set_x, set_y = set_y, set_x
        m, n = n, m

    dist_matrix = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            dist = np.abs(set_x[i] - set_y[j])
            dist_matrix[i, j] = min(int(dist), c) ** p

    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    total_cost = dist_matrix[row_ind, col_ind].sum()

    if p == np.inf:
        if m == n:
            ospa_dist = np.max(dist_matrix[row_ind, col_ind])
        else:
            ospa_dist = c
    else:
        cardinality_error = c**p * (n - m)
        ospa_dist = (1 / n * (total_cost + cardinality_error)) ** (1 / p)

    return ospa_dist


if __name__ == "__main__":
    x = np.array([-26, 0, 34, 67])
    y = np.array([0, 67, 34])
    print("Hausdorff distance: {}".format(hausdorff_distance(x, y)))
    print("OSPA distance: {}".format(ospa_distance(x, y)))
