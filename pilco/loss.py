import autograd.numpy as np
from autograd.numpy import exp, sqrt
from autograd.numpy.linalg import solve, det


class loss:
    def loss_sat(self, m, s):
        m = m.reshape(-1, 1)
        D, _ = m.shape

        W = self.W if hasattr(self, 'W') else np.eye(D)
        z = self.z if hasattr(self, 'z') else np.zeros([D, 1])

        sW = np.dot(s, W)
        ispW = solve((np.eye(D) + sW).T, W.T).T
        L = -exp(-(m - z).T @ ispW @ (m - z) / 2) / sqrt(det(np.eye(D) + sW))

        i2spW = solve((np.eye(D) + 2 * sW).T, W.T).T
        r2 = exp(-(m - z).T @ i2spW @ (m - z)) / sqrt(det(np.eye(D) + 2 * sW))
        S = r2 - L**2
        if S < 1e-12:
            S = 0

        t = np.dot(W, z) - ispW @ (np.dot(sW, z) + m)
        C = L * t

        return L + 1, S, C
