import autograd.numpy as np
from autograd import value_and_grad
from autograd.numpy import exp, log, sqrt, std, newaxis
from autograd.numpy.linalg import solve, cholesky, det
from scipy.optimize import minimize

from pilco import empty


def maha(a, b, Q):
    aQ = np.matmul(a, Q)
    bQ = np.matmul(b, Q)
    K = np.expand_dims(np.sum(aQ * a, -1), -1) + np.expand_dims(
        np.sum(bQ * b, -1), -2) - 2 * np.einsum('...ij, ...kj->...ik', aQ, b)
    return K


class kernel:
    def __init__(self):
        pass

    def __add__(self, other):
        sum = kernel()
        sum.sub = self, other
        sum.num_hyp = lambda x: self.num_hyp(x) + other.num_hyp(x)
        return sum

    def __call__(self, loghyp, x, z=None):
        loghyp = np.atleast_2d(loghyp)
        left, right = self.sub
        L = left.num_hyp(x)
        return left(loghyp[:, :L], x, z) + right(loghyp[:, L:], x, z)


class kernel_rbf(kernel):
    """
    Squared Exponential covariance function.
    """

    def __init__(self):
        super().__init__()
        self.num_hyp = lambda x: np.size(x, 1) + 1

    def __call__(self, loghyp, x, z=None):
        loghyp = np.atleast_2d(loghyp)
        n, D = x.shape
        ell = exp(loghyp[:, :D])  # [E, D]
        sf2 = exp(2 * loghyp[:, D])
        sf2 = sf2.reshape(-1, 1, 1)

        x_ell = np.expand_dims(x, 0) / np.expand_dims(ell, 1)  # [E, n, D]
        if z is None:
            diff = np.expand_dims(x_ell, 1) - np.expand_dims(x_ell, 2)
        else:
            z_ell = np.expand_dims(z, 0) / np.expand_dims(ell, 1)
            diff = np.expand_dims(x_ell, 1) - np.expand_dims(z_ell, 2)

        K = sf2 * exp(-0.5 * np.sum(diff**2, axis=3))  # [E, n, n]
        return K


class kernel_noise(kernel):
    """
    White noise.
    """

    def __init__(self):
        super().__init__()
        self.num_hyp = lambda x: 1

    def __call__(self, loghyp, x, z=None):
        loghyp = np.atleast_2d(loghyp)
        n, D = x.shape
        s2 = np.exp(2 * loghyp)  # [E, 1]
        s2 = s2.reshape(-1, 1, 1)

        if z is None:
            K = s2 * np.expand_dims(np.eye(n), 0)
        else:
            K = 0
        return K


class gpmodel:
    def __init__(self, kernel=None):
        if kernel is not None:
            self.kernel = kernel
        else:
            self.kernel = kernel_rbf() + kernel_noise()

    def log_pdf(self, hyp):
        x = np.atleast_2d(self.inputs)
        y = np.atleast_2d(self.targets)

        n, D = x.shape
        n, E = y.shape

        hyp = hyp.reshape(E, -1)
        K = self.kernel(hyp, x)  # [E, n, n]
        L = cholesky(K)
        alpha = np.hstack([solve(K[i], y[:, i]) for i in range(E)])
        y = y.flatten(order='F')

        logp = 0.5 * n * E * log(2 * np.pi) + 0.5 * np.dot(y, alpha) + np.sum(
            [log(np.diag(L[i])) for i in range(E)])

        return logp

    def hyp_crub(self, hyp):
        x = np.atleast_2d(self.inputs)
        y = np.atleast_2d(self.targets)

        n, D = x.shape
        n, E = y.shape
        hyp = hyp.reshape(E, -1)
        p = 30

        if np.size(hyp, 1) == 3 * D + 2:
            ll = hyp[:, :2 * D]
            lsf = hyp[:, 2 * D:-1]
        elif np.size(hyp, 1) == 2 * D + 1:
            ll = hyp[:, :D]
            lsf = hyp[:, D:-1]
        elif np.size(hyp, 1) == D + 2:
            ll = hyp[:, :D]
            lsf = hyp[:, D]
        else:
            raise ValueError('Incorrect number of hyperparameters.')
        lsn = hyp[:, -1]

        L = self.log_pdf(hyp)
        L = L + np.sum(((ll - log(self.curb.std)) / log(self.curb.ls))**p)
        L = L + np.sum(((lsf - lsn) / log(self.curb.snr))**p)
        return L

    def train(self, curb=None):
        assert hasattr(self, "inputs")
        assert hasattr(self, "targets")
        x = np.atleast_2d(self.inputs)
        y = np.atleast_2d(self.targets)
        assert len(x) == len(y)

        n, D = x.shape
        n, E = y.shape

        if curb is not None:
            self.curb = curb
        elif not hasattr(self, "curb"):
            self.curb = empty()
            self.curb.snr = 500
            self.curb.ls = 100
            self.curb.std = std(x, 0)

        if not hasattr(self, "hyp"):
            self.hyp = np.zeros([E, D + 2])
            self.hyp[:, :D] = np.repeat(log(std(x, 0)).reshape(1, D), E, 0)
            self.hyp[:, D] = log(std(y, 0))
            self.hyp[:, -1] = log(std(y, 0) / 10)

        print("Train hyperparameters of full GP...")
        try:
            self.result = minimize(
                value_and_grad(self.hyp_crub), self.hyp, jac=True)
        except Exception:
            self.result = minimize(
                value_and_grad(self.hyp_crub), self.hyp, jac=True, method='CG')

        self.hyp = self.result.get('x').reshape(E, -1)
        self.K = self.kernel(self.hyp, x)
        self.iK = np.stack([solve(self.K[i], np.eye(n)) for i in range(E)])
        self.alpha = np.vstack([solve(self.K[i], y[:, i]) for i in range(E)]).T

    def gp0(self, m, s):
        """
        Compute joint predictions for MGP with uncertain inputs.
        """
        assert hasattr(self, "hyp")
        assert hasattr(self, "iK")

        x = np.atleast_2d(self.inputs)
        y = np.atleast_2d(self.targets)
        n, D = x.shape
        n, E = y.shape

        X = self.hyp
        iK = self.iK
        beta = self.alpha

        m = np.atleast_2d(m)
        inp = x - m

        # Compute the predicted mean and IO covariance.
        iL = np.stack([np.diag(exp(-X[i, :D])) for i in range(E)])
        iN = np.matmul(inp, iL)
        B = iL @ s @ iL + np.eye(D)
        t = np.stack([solve(B[i].T, iN[i].T).T for i in range(E)])
        q = exp(-np.sum(iN * t, 2) / 2)
        qb = q * beta.T
        tiL = np.matmul(t, iL)
        c = exp(2 * X[:, D]) / sqrt(det(B))

        M = np.sum(qb, 1) * c
        V = (np.transpose(tiL, [0, 2, 1]) @ np.expand_dims(qb, 2)).reshape(
            E, D) * c.reshape(E, 1)
        k = 2 * X[:, D].reshape(E, 1) - np.sum(iN**2, 2) / 2

        # Compute the predicted covariance.
        inp = np.expand_dims(inp, 0) / np.expand_dims(exp(2 * X[:, :D]), 1)
        ii = np.broadcast_to(inp[:, newaxis, :, :], (E, E, n, D))
        ij = np.broadcast_to(inp[newaxis, :, :, :], (E, E, n, D))

        iL = np.stack([np.diag(exp(-2 * X[i, :D])) for i in range(E)])
        siL = np.expand_dims(iL, 0) + np.expand_dims(iL, 1)
        R = np.matmul(s, siL) + np.eye(D)
        t = 1 / sqrt(det(R))
        iRs = np.stack(
            [solve(R.reshape(-1, D, D)[i], s) for i in range(E * E)])
        iRs = iRs.reshape(E, E, D, D)
        Q = exp(k[:, newaxis, :, newaxis] + k[newaxis, :, newaxis, :] +
                maha(ii, -ij, iRs / 2))

        S = np.einsum('ji,iljk,kl->il', beta, Q, beta)
        tr = np.hstack([np.sum(Q[i, i] * iK[i]) for i in range(E)])
        S = (S - np.diag(tr)) * t + np.diag(exp(2 * X[:, D]))
        S = S - np.matmul(M[:, newaxis], M[newaxis, :])

        return M, S, V

    def gp2(self, m, s):
        assert hasattr(self, "hyp")

        x = np.atleast_2d(self.inputs)
        y = np.atleast_2d(self.targets)
        n, D = x.shape
        n, E = y.shape

        X = self.hyp
        beta = self.alpha

        m = np.atleast_2d(m)
        inp = x - m

        # Compute the predicted mean and IO covariance.
        iL = np.stack([np.diag(exp(-X[i, :D])) for i in range(E)])
        iN = np.matmul(inp, iL)
        B = iL @ s @ iL + np.eye(D)
        t = np.stack([solve(B[i].T, iN[i].T).T for i in range(E)])
        q = exp(-np.sum(iN * t, 2) / 2)
        qb = q * beta.T
        tiL = np.matmul(t, iL)
        c = exp(2 * X[:, D]) / sqrt(det(B))

        M = np.sum(qb, 1) * c
        V = (np.transpose(tiL, [0, 2, 1]) @ np.expand_dims(qb, 2)).reshape(
            E, D) * c.reshape(E, 1)
        k = 2 * X[:, D].reshape(E, 1) - np.sum(iN**2, 2) / 2

        # Compute the predicted covariance.
        inp = np.expand_dims(inp, 0) / np.expand_dims(exp(2 * X[:, :D]), 1)
        ii = np.broadcast_to(inp[:, newaxis, :, :], (E, E, n, D))
        ij = np.broadcast_to(inp[newaxis, :, :, :], (E, E, n, D))

        iL = np.stack([np.diag(exp(-2 * X[i, :D])) for i in range(E)])
        siL = np.expand_dims(iL, 0) + np.expand_dims(iL, 1)
        R = np.matmul(s, siL) + np.eye(D)
        t = 1 / sqrt(det(R))
        iRs = np.stack(
            [solve(R.reshape(-1, D, D)[i], s) for i in range(E * E)])
        iRs = iRs.reshape(E, E, D, D)
        Q = exp(k[:, newaxis, :, newaxis] + k[newaxis, :, newaxis, :] +
                maha(ii, -ij, iRs / 2))

        S = t * np.einsum('ji,iljk,kl->il', beta, Q, beta) + 1e-6 * np.eye(E)
        S = S - np.matmul(M[:, newaxis], M[newaxis, :])

        return M, S, V
