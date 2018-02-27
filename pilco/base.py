import autograd.numpy as np
from autograd.numpy import newaxis
from autograd.numpy.random import rand, multivariate_normal
from scipy.integrate import odeint
from scipy.optimize import minimize
from autograd import value_and_grad

from pilco.util import gaussian_trig, fill_mat, unwrap, rewrap


def rollout(start, policy, plant, cost, H):
    """
    Generate a state trajectory using an ODE solver.
    """
    odei = plant.odei
    poli = plant.poli
    dyno = plant.dyno
    angi = plant.angi

    nX = len(odei)
    nU = len(policy.max_u)
    nA = len(angi)

    state = start
    x = np.zeros([H + 1, nX + 2 * nA])
    x[0, odei] = multivariate_normal(start, plant.noise)

    u = np.zeros([H, nU])
    y = np.zeros([H, nX])
    L = np.zeros(H)
    latent = np.zeros([H + 1, nX + nU])

    for i in range(H):
        s = x[i, odei]
        a, _, _ = gaussian_trig(s, 0 * np.eye(nX), angi)
        s = np.hstack([s, a])
        x[i, -2 * nA:] = s[-2 * nA:]

        if hasattr(policy, "fcn"):
            u[i, :], _, _ = policy.fcn(s[poli], 0 * np.eye(len(poli)))
        else:
            u[i, :] = policy.max_u * (2 * rand(nU) - 1)
        latent[i, :] = np.hstack([state, u[i, :]])

        dynamics = plant.dynamics
        dt = plant.dt
        next = odeint(dynamics, state[odei], [0, dt], args=(u[i, :], ))
        state = next[-1, :]
        x[i + 1, odei] = multivariate_normal(state[odei], plant.noise)

        if hasattr(cost, "fcn"):
            L[i] = cost.fcn(cost, state[dyno], 0 * np.eye(len(dyno)))

    y = x[1:H + 1, :nX]
    x = np.hstack([x[:H, :], u[:H, :]])
    latent[H, :nX] = state

    return x, y, L, latent


def train(gpmodel, plant, policy, x, y):
    Du = len(policy.max_u)
    dyni = np.asarray(plant.dyni)
    dyno = np.asarray(plant.dyno)
    difi = np.asarray(plant.difi)

    gpmodel.inputs = np.hstack([x[:, dyni], x[:, -Du:]])
    gpmodel.targets = y[:, dyno]
    gpmodel.targets[:, difi] = gpmodel.targets[:, difi] - x[:, dyno[difi]]
    gpmodel.optimize()

    hyp = gpmodel.hyp
    print(gpmodel.result['message'])
    print("Learned noise std:\n%s" % (str(np.exp(hyp[:, -1]))))
    print("SNRs:\n%s" % (str(np.exp(hyp[:, -2] - hyp[:, -1]))))


def propagate(m, s, plant, dynmodel, policy):
    angi = plant.angi
    poli = plant.poli
    dyni = plant.dyni
    difi = plant.difi

    D0 = len(m)
    D1 = D0 + 2 * len(angi)
    D2 = D1 + len(policy.max_u)
    M = np.array(m)
    S = s

    i, j = np.arange(D0), np.arange(D0, D1)
    m, s, c = gaussian_trig(M[i], S[np.ix_(i, i)], angi)
    q = np.matmul(S[np.ix_(i, i)], c)
    M = np.hstack([M, m])
    S = np.vstack([np.hstack([S, q]), np.hstack([q.T, s])])

    i, j = poli, np.arange(D1)
    m, s, c = policy.fcn(M[i], S[np.ix_(i, i)])
    q = np.matmul(S[np.ix_(j, i)], c)
    M = np.hstack([M, m])
    S = np.vstack([np.hstack([S, q]), np.hstack([q.T, s])])

    i, j = np.hstack([dyni, np.arange(D1, D2)]), np.arange(D2)
    m, s, c = dynmodel.fcn(M[i], S[np.ix_(i, i)])
    q = np.matmul(S[np.ix_(j, i)], c)
    M = np.hstack([M, m])
    S = np.vstack([np.hstack([S, q]), np.hstack([q.T, s])])

    P = np.hstack([np.zeros((D0, D2)), np.eye(D0)])
    P = fill_mat(np.eye(len(difi)), P, difi, difi)
    M_next = np.matmul(P, M[:, newaxis]).flatten()
    S_next = P @ S @ P.T
    S_next = (S_next + S_next.T) / 2
    return M_next, S_next


def value(p, mu0, S0, dynmodel, policy, plant, cost, H):
    policy.p = rewrap(p, policy)
    M = mu0
    S = S0
    L = np.array([0])

    for t in range(H):
        M, S = plant.prop(M, S, plant, dynmodel, policy)
        L = L + cost.gamma**t * cost.fcn(cost, M, S)
    return L


def learn(mu0, S0, dynmodel, policy, plant, cost, H):
    def value_bind(p):
        return value(p, mu0, S0, dynmodel, policy, plant, cost, H)

    def callback(p):
        print(np.asscalar(value(p, mu0, S0, dynmodel, policy, plant, cost, H)))

    options = {'maxiter': 150, 'disp': True}
    result = minimize(
        value_and_grad(value_bind),
        unwrap(policy.p),
        jac=True,
        options=options,
        callback=callback)

    policy.p = rewrap(result['x'], policy)
