import autograd.numpy as np
from autograd.numpy import sin, cos, exp

__all__ = ['Empty']


class Empty(dict):
    pass


def fill_mat(m, n, i=None, j=None):
    """
    Fill a matrix m of size [a, b] into a larger one [p, q],
    according to given indices i, j.
    """
    m, n = np.atleast_2d(m), np.atleast_2d(n)
    a, b = m.shape
    p, q = n.shape
    i = np.arange(a) if i is None else np.atleast_1d(i)
    j = np.arange(b) if j is None else np.atleast_1d(j)

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
    return Ti @ m @ Tj + n


def gaussian_trig(m, v, i, e=None):
    d = len(m)
    L = len(i)
    e = np.ones((1, L)) if e is None else np.atleast_2d(e)
    ee = np.vstack([e, e]).reshape(1, -1, order='F')

    mi = np.atleast_2d(m[i])
    vi = v[np.ix_(i, i)]
    vii = np.atleast_2d(np.diag(vi))

    M = np.vstack([e * exp(-vii / 2) * sin(mi), e * exp(-vii / 2) * cos(mi)])
    M = M.flatten(order='F')

    lq = -(vii.T + vii) / 2
    q = exp(lq)

    U1 = (exp(lq + vi) - q) * sin(mi.T - mi)
    U2 = (exp(lq - vi) - q) * sin(mi.T + mi)
    U3 = (exp(lq + vi) - q) * cos(mi.T - mi)
    U4 = (exp(lq - vi) - q) * cos(mi.T + mi)

    V = np.vstack(
        [np.hstack([U3 - U4, U1 + U2]),
         np.hstack([(U1 + U2).T, U3 + U4])])
    V = np.vstack([
        np.hstack([V[::2, ::2], V[::2, 1::2]]),
        np.hstack([V[1::2, ::2], V[1::2, 1::2]])
    ])
    V = np.dot(ee.T, ee) * V / 2

    C = np.hstack([np.diag(M[1::2]), -np.diag(M[::2])])
    C = np.hstack([C[:, ::2], C[:, 1::2]])
    C = fill_mat(C, np.zeros((d, 2 * L)), i, None)

    return M, V, C


def gaussian_sin(m, v, i, e=None):
    d = len(m)
    L = len(i)
    e = np.ones((1, L)) if e is None else np.atleast_2d(e)

    mi = np.atleast_2d(m[i])
    vi = v[np.ix_(i, i)]
    vii = np.atleast_2d(np.diag(vi))
    M = e * exp(-vii / 2) * sin(mi)
    M = M.flatten()

    lq = -(vii.T + vii) / 2
    q = exp(lq)
    V = ((exp(lq + vi) - q) * cos(mi.T - mi) -
         (exp(lq - vi) - q) * cos(mi.T + mi))
    V = np.dot(e.T, e) * V / 2

    C = np.diag((e * exp(-vii / 2) * cos(mi)).flatten())
    C = fill_mat(C, np.zeros((d, L)), i, None)

    return M, V, C


def maha(a, b, Q):
    aQ = np.matmul(a, Q)
    bQ = np.matmul(b, Q)
    K = np.expand_dims(np.sum(aQ * a, -1), -1) + np.expand_dims(
        np.sum(bQ * b, -1), -2) - 2 * np.einsum('...ij, ...kj->...ik', aQ, b)
    return K


def unwrap(p):
    return np.hstack([v.flatten() for v in p.values()])


def rewrap(m, p):
    d = {}
    start = 0
    for k, v in p.items():
        length = np.size(v)
        d[k] = np.reshape(m[start:start + length], v.shape)
        start = start + length
    return d
