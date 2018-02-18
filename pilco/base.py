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

    y = x[1:H + 1, 0:nX]
    x = np.hstack([x[0:H, :], u[0:H, :]])
    latent[H, 0:nX] = state

    return x, y, L, latent
