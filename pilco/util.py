import autograd.numpy as np
from autograd.numpy import sin, cos, exp


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
    T = np.zeros([d, L])
    for j, k in enumerate(i):
        T[k, j] = 1
    C = np.dot(T, C)

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
    T = np.zeros([d, L])
    for j, k in enumerate(i):
        T[k, j] = 1
    C = np.dot(T, C)

    return M, V, C
