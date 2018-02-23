import autograd.numpy as np
from autograd.numpy.random import rand, multivariate_normal
from scipy.integrate import odeint

from pilco.util import gaussian_trig


def rollout(start, policy, H, plant, cost):
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
            u[i, :] = policy.fcn(s[poli], 0 * np.eye(len(poli)))
        else:
            u[i, :] = policy.max_u * (2 * rand(nU) - 1)
        latent[i, :] = np.hstack([state, u[i, :]])

        dynamics = plant.dynamics
        dt = plant.dt
        next = odeint(dynamics, state[odei], [0, dt], args=(u[i, :], ))
        state = next[-1, :]
        x[i + 1, odei] = multivariate_normal(state[odei], plant.noise)

        if hasattr(cost, "fcn"):
            L[i] = cost.fcn(state[dyno], 0 * np.eye(len(dyno)))

    y = x[1:H + 1, :nX]
    x = np.hstack([x[:H, :], u[:H, :]])
    latent[H, :nX] = state

    return x, y, L, latent


def train_dynmodel(gpmodel, plant, policy, x, y):
    Du = len(policy.max_u)
    dyni = np.asarray(plant.dyni)
    dyno = np.asarray(plant.dyno)
    difi = np.asarray(plant.difi)

    gpmodel.inputs = np.hstack([x[:, dyni], x[:, -Du:]])
    gpmodel.targets = y[:, dyno]
    gpmodel.targets[:, difi] = gpmodel.targets[:, difi] - x[:, dyno[difi]]
    gpmodel.train()

    hyp = gpmodel.hyp
    print(gpmodel.result)
    print("Learned noise std:\n%s" % (str(np.exp(hyp[:, -1]))))
    print("SNRs:\n%s" % (str(np.exp(hyp[:, -2] - hyp[:, -1]))))


def propagate(gpmodel, plant, policy, m, s):
    pass
