import autograd.numpy as np
from autograd.numpy import sin, cos, exp

__all__ = ['empty']


class empty(dict):
    pass


def fill_to(m, shape, i=None, j=None):
    """
    Fill a matrix m of size [a, b] into a larger one [p, q],
    according to given indices i, j.
    """
    a, b = m.shape
    p, q = shape
    i = np.arange(a) if i is None else np.array(i)
    j = np.arange(b) if j is None else np.array(j)

    if a > p or b > q:
        raise ValueError("Shape error!")
    if len(i) != a or len(j) != b:
        raise ValueError("Indices not match!")
    if not (max(i) < p and max(j) < q):
        raise ValueError("Indices out of bound!")

    Ti = np.zeros((p, a))
    Tj = np.zeros((b, q))
    for u, v in enumerate(i):
        Ti[v, u] = 1
    for u, v in enumerate(j):
        Tj[u, v] = 1
    return Ti @ m @ Tj


def gaussian_trig(m, v, i):
    d = len(m)
    L = len(i)
    mi = m[i]
    vi = v[np.ix_(i, i)]
    vii = np.diag(vi)

    M = np.vstack([exp(-vii / 2) * sin(mi), exp(-vii / 2) * cos(mi)])
    M = M.flatten(order='F')

    mi = mi.reshape(L, 1)
    vii = vii.reshape(L, 1)
    lq = -(vii + vii.T) / 2
    q = exp(lq)

    U1 = (exp(lq + vi) - q) * sin(mi - mi.T)
    U2 = (exp(lq - vi) - q) * sin(mi + mi.T)
    U3 = (exp(lq + vi) - q) * cos(mi - mi.T)
    U4 = (exp(lq - vi) - q) * cos(mi + mi.T)

    V = np.vstack(
        [np.hstack([U3 - U4, U1 + U2]),
         np.hstack([(U1 + U2).T, U3 + U4])])
    V = np.vstack([
        np.hstack([V[::2, ::2], V[::2, 1::2]]),
        np.hstack([V[1::2, ::2], V[1::2, 1::2]])
    ])
    V = V / 2

    C = np.hstack([np.diag(M[1::2]), -np.diag(M[::2])])
    C = np.hstack([C[:, ::2], C[:, 1::2]])
    C = fill_to(C, (d, 2 * L), i, None)

    return M, V, C


def gaussian_sin(m, v, i):
    d = len(m)
    L = len(i)
    mi = m[i]
    vi = v[np.ix_(i, i)]
    vii = np.diag(vi)
    M = exp(-vii / 2) * sin(mi)

    mi = mi.reshape(L, 1)
    vii = vii.reshape(L, 1)
    lq = -(vii + vii.T) / 2
    q = exp(lq)
    V = ((exp(lq + vi) - q) * cos(mi - mi.T) -
         (exp(lq - vi) - q) * cos(mi + mi.T))
    V = V / 2

    C = np.diag((exp(-vii / 2) * cos(mi)).flatten())
    C = fill_to(C, (d, L), i, None)

    return M, V, C


def maha(a, b, Q):
    aQ = np.matmul(a, Q)
    bQ = np.matmul(b, Q)
    K = np.expand_dims(np.sum(aQ * a, -1), -1) + np.expand_dims(
        np.sum(bQ * b, -1), -2) - 2 * np.einsum('...ij, ...kj->...ik', aQ, b)
    return K
