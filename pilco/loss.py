import autograd.numpy as np
from autograd.numpy import exp, sqrt
from autograd.numpy.linalg import solve, det


def loss_sat(cost, m, s):
    D = len(m)

    W = cost.W if hasattr(cost, 'W') else np.eye(D)
    z = cost.z if hasattr(cost, 'z') else np.zeros(D)
    m, z = np.atleast_2d(m), np.atleast_2d(z)

    sW = np.dot(s, W)
    ispW = solve((np.eye(D) + sW).T, W.T).T
    L = -exp(-(m - z) @ ispW @ (m - z).T / 2) / sqrt(det(np.eye(D) + sW))

    i2spW = solve((np.eye(D) + 2 * sW).T, W.T).T
    r2 = exp(-(m - z) @ i2spW @ (m - z).T) / sqrt(det(np.eye(D) + 2 * sW))
    S = r2 - L**2

    t = np.dot(W, z.T) - ispW @ (np.dot(sW, z.T) + m.T)
    C = L * t

    return L + 1, S, C
