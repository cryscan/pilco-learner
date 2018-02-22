import autograd.numpy as np
from autograd import value_and_grad
from autograd.numpy import exp, log, sqrt, std
from autograd.numpy.linalg import solve, cholesky, det
from scipy.optimize import minimize

from pilco import empty


def maha(a, b, Q):
    _, E, n, D = a.shape
    E, _, n, D = b.shape
    E, E, D, D = Q.shape
    a = np.broadcast_to(a, [E, E, n, D])
    b = np.broadcast_to(b, [E, E, n, D])

    aQ = a @ Q
    K = np.expand_dims(np.sum(aQ * a, -1), -2) + np.expand_dims(
        np.sum(b @ Q * b, -1), -1) - 2 * aQ @ np.transpose(b, [0, 1, 3, 2])
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
    """
    Before training:
    kernel
    crub

    hyp      [E, D + 2]
    inputs   [n, D]
    targets  [n, E]

    After training:
    beta     [n, E]
    K        [n, n]
    iK       [n, n]
    """

    def __init__(self, kernel=None):
        if kernel is not None:
            self.kernel = kernel
        else:
            self.kernel = kernel_rbf() + kernel_noise()

    def _log_pdf(self, hyp):
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

    def _hyp_crub(self, hyp):
        x = np.atleast_2d(self.inputs)
        y = np.atleast_2d(self.targets)

        n, D = x.shape
        n, E = y.shape
        hyp = hyp.reshape(E, -1)
        p = 30

        if np.size(hyp, 1) == 3 * D + 2:
            ll = hyp[:, :2 * D]
            lsf = hyp[:, 2 * D:3 * D + 1]
        elif np.size(hyp, 1) == 2 * D + 1:
            ll = hyp[:, :D]
            lsf = hyp[:, D:2 * D]
        elif np.size(hyp, 1) == D + 2:
            ll = hyp[:, :D]
            lsf = hyp[:, D + 1]
        else:
            raise ValueError('Incorrect number of hyperparameters.')
        lsn = hyp[:, -1]

        L = self._log_pdf(hyp)
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
                value_and_grad(self._hyp_crub), self.hyp, jac=True)
        except Exception:
            self.result = minimize(
                value_and_grad(self._hyp_crub),
                self.hyp,
                jac=True,
                method='CG')

        print(self.result)
        self.hyp = self.result.get('x').reshape(E, -1)
        self.K = self.kernel(self.hyp, x)
        self.iK = np.stack([solve(self.K[i], np.eye(n)) for i in range(E)])
        self.alpha = np.vstack([solve(self.K[i], y[:, i]) for i in range(E)]).T

    def gp0(self, m, s):
        """
        Compute joint predictions for MGP with uncertain inputs.

        Input arguments:
        m  [1, D]
        s  [D, D]

        Output arguments:
        M  [1, E]
        S  [E, E]
        V  [E, D]
        """
        assert hasattr(self, "hyp")
        assert hasattr(self, "K")

        x = np.atleast_2d(self.inputs)
        y = np.atleast_2d(self.targets)
        n, D = x.shape
        n, E = y.shape

        X = self.hyp  # [E, D + 2]
        iK = self.iK  # [n, n]
        beta = self.alpha

        m = m.reshape(1, D)
        inp = x - m  # [n, D]

        # Compute the predicted mean and IO covariance.
        iL = np.stack([np.diag(exp(-X[i, :D])) for i in range(E)])  # [E, D, D]
        iN = inp @ iL  # [E, n, D]
        B = iL @ s @ iL + np.eye(D)  # [E, D, D]
        # t = iN @ inv(B)
        t = np.stack([solve(B[i].T, iN[i].T).T for i in range(E)])  # [E, n, D]
        q = exp(-np.sum(iN * t, 2) / 2)  # [E, n]
        qb = q * beta.T  # [E, n]
        tiL = t @ iL  # [E, n, D]
        c = exp(2 * X[:, D]) / sqrt(det(B))  # [E]

        M = (np.sum(qb, 1) * c).reshape(1, E)
        V = (np.transpose(tiL, [0, 2, 1]) @ np.expand_dims(qb, 2)).reshape(
            E, D) * c.reshape(E, 1)
        k = 2 * X[:, D].reshape(E, 1) - np.sum(iN**2, 2) / 2  # [E, n]

        # Compute the predicted covariance.
        # [E, n, D]
        ii = np.expand_dims(inp, 0) / np.expand_dims(exp(2 * X[:, :D]), 1)
        # [E, D, D]
        iL = np.stack([np.diag(exp(-2 * X[i, :D])) for i in range(E)])
        siL = np.expand_dims(iL, 0) + np.expand_dims(iL, 1)  # [E, E, D, D]
        R = s @ siL + np.eye(D)  # [E, E, D, D]
        t = 1 / sqrt(det(R))  # [E, E]
        iRs = np.stack([
            solve(R.reshape(E * E, D, D)[i], s) for i in range(E * E)
        ]).reshape(E, E, D, D)
        # [E, E, n, n]
        L = exp(k.reshape(E, 1, n, 1) + k.reshape(1, E, 1, n)) + maha(
            np.expand_dims(ii, 0), -np.expand_dims(ii, 1), iRs / 2)

        S = np.einsum('ji,iljk,kl->il', beta, L, beta)  # [E, E]
        tr = np.hstack([np.sum(L[i, i] * iK[i]) for i in range(E)])
        S = (S - np.diag(tr)) * t + np.diag(exp(2 * X[:, D])) - M.T @ M

        return M, S, V
