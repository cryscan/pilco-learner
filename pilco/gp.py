import autograd.numpy as np
from autograd import value_and_grad
from autograd.numpy import exp, log, std, linalg
from scipy.optimize import minimize

from pilco import empty


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

    def _log_pdf(self, hyp):
        x = np.atleast_2d(self.inputs)
        y = np.atleast_2d(self.targets)

        n, D = x.shape
        n, E = y.shape

        hyp = hyp.reshape(E, -1)
        K = self.kernel(hyp, x)  # [E, n, n]
        alpha = np.hstack([linalg.solve(K[i], y[:, i]) for i in range(E)])
        L = linalg.cholesky(K)

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
            self.curb.std = std(x, axis=0)

        if not hasattr(self, "hyp"):
            self.hyp = np.zeros([E, D + 2])
            self.hyp[:, :D] = np.repeat(
                log(std(x, axis=0)).reshape(1, D), E, axis=0)
            self.hyp[:, D] = log(std(y, axis=0))
            self.hyp[:, -1] = log(std(y, axis=0) / 10)

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
        alpha = np.vstack([linalg.solve(self.K[i], y[:, i]) for i in range(E)])
        self.alpha = alpha.T
