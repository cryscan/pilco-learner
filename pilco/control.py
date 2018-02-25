import autograd.numpy as np
from autograd.numpy import log

from pilco.util import fill_mat


def congp(policy, m, s):
    policy.hyp = np.atleast_2d(policy.hyp)
    policy.inputs = np.atleast_2d(policy.inputs)
    policy.targets = np.atleast_2d(policy.targets)

    E, D = policy.hyp.shape
    T = np.zeros_like(policy.hyp)
    T[:, (-2, -1)] = np.broadcast_to([log(1), log(0.01)], (E, 2))
    policy.hyp = (T == 0) * policy.hyp + T

    return policy.gp2(m, s)


def concat(con, sat, policy, m, s):
    max_u = policy.max_u
    E = len(max_u)
    D = len(m)

    F = D + E
    i, j = np.arange(D), np.arange(D, F)
    M = m
    S = fill_mat(s, np.zeros((F, F)))

    m, s, c = con(policy, m, s)
    M = np.hstack([M, m])
    S = fill_mat(s, S, j, j)
    q = np.matmul(S[np.ix_(i, i)], c)
    S = fill_mat(q, S, i, j)
    S = fill_mat(q.T, S, j, i)

    M, S, R = sat(M, S, j, max_u)
    C = np.hstack([np.eye(D), c]) @ R
    return M, S, C
