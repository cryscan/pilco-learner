import autograd.numpy as np
from autograd.numpy import random, linalg
from autograd.scipy.stats import multivariate_normal
from autograd import value_and_grad
from scipy.optimize import minimize


class kernel:
    def __init__(self, loghyp):
        self.loghyp = np.atleast_2d(loghyp)
        self.E = len(self.loghyp)

    def __add__(self, other):
        if self.E != other.E:
            raise ValueError("Target dimensions not match!", self.E)
        sum = kernel(np.hstack([self.loghyp, other.loghyp]))
        sum.component = self, other
        return sum

    def __call__(self, x, z=None):
        if hasattr(self, "component"):
            left, right = self.component
            return left(x, z) + right(x, z)
        else:
            raise NotImplementedError


class kernel_rbf(kernel):
    """
    Squared Exponential covariance function.
    """

    def __init__(self, loghyp):
        super().__init__(loghyp)

    def __call__(self, x, z=None):
        n, D = x.shape
        ell = np.exp(self.loghyp[:, :D])  # [E, D]
        sf2 = np.exp(2 * self.loghyp[:, D])
        sf2 = sf2.reshape(self.E, 1, 1)

        x_ell = np.expand_dims(x, 0) / np.expand_dims(ell, 1)  # [E, n, D]

        if z is None:
            diff = np.expand_dims(x_ell, 1) - np.expand_dims(x_ell, 2)
        else:
            z_ell = np.expand_dims(z, 0) / np.expand_dims(ell, 1)
            diff = np.expand_dims(x_ell, 1) - np.expand_dims(z_ell, 2)

        K = sf2 * np.exp(np.sum(diff**2, axis=3))  # [E, n, n]
        return K


class kernel_noise(kernel):
    """
    White noise.
    """

    def __init__(self, loghyp):
        super().__init__(loghyp)

    def __call__(self, x, z=None):
        n, D = x.shape
        s2 = np.exp(2 * self.loghyp)  # [E, 1]
        s2 = s2.reshape(self.E, 1, 1)

        if z is None:
            K = s2 * np.expand_dims(np.eye(n), 0)
        else:
            K = 0
        return K
