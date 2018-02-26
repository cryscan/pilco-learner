import math

import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd.numpy import sin, cos, log
from autograd.numpy.random import randn, multivariate_normal
from matplotlib import animation

from pilco import empty
from pilco.base import rollout, train, propagate
from pilco.util import gaussian_trig, gaussian_sin
from pilco.gp import gpmodel
from pilco.control import congp, concat


def dynamics(z, t, u):
    g = 9.82
    L = 0.6
    m1 = 0.5
    m2 = 0.5
    b = 0.1

    z1, z2, z3 = z[1], z[2], z[3]
    d = 4 * (m1 + m2) - 3 * m1 * cos(z3)**2

    dzdt = np.zeros_like(z)
    dzdt[0] = z1
    dzdt[1] = (2 * m1 * L * z2**2 * sin(z3) + 3 * m1 * g * sin(z3) * cos(z3) +
               4 * u - 4 * b * z1) / d
    dzdt[2] = (-3 * m1 * L * z2**2 * sin(z3) * cos(z3) - 6 *
               (m1 + m2) * g * sin(z3) - 6 * (u - b * z1) * cos(z3)) / (d * L)
    dzdt[3] = z2

    return dzdt


odei = [0, 1, 2, 3]
dyno = [0, 1, 2, 3]
angi = [3]
dyni = [0, 1, 2, 4, 5]
poli = [0, 1, 2, 4, 5]
difi = [0, 1, 2, 3]

dt = 0.1
T = 4
H = math.ceil(T / dt)
mu0 = np.array([0, 0, 0, 0])
S0 = np.square(np.diag([0.1, 0.1, 0.1, 0.1]))
N = 15
nc = 10

plant = empty()
plant.dynamics = dynamics
plant.prop = propagate
plant.noise = np.square(np.diag([1e-2, 1e-2, 1e-2, 1e-2]))
plant.dt = dt
plant.odei = odei
plant.angi = angi
plant.poli = poli
plant.dyno = dyno
plant.dyni = dyni
plant.difi = difi

m, s, c = gaussian_trig(mu0, S0, angi)
m = np.hstack([mu0, m])
c = np.dot(S0, c)
s = np.vstack([np.hstack([S0, c]), np.hstack([c.T, s])])

policy = gpmodel()
policy.max_u = [10]

p = {
    'inputs': multivariate_normal(m[poli], s[np.ix_(poli, poli)], nc),
    'targets': 0.1 * randn(nc, len(policy.max_u)),
    'hyp': log([1, 1, 1, 0.7, 0.7, 1, 0.01])
}
policy.p = p

cost = empty()

x, y, _, latent = rollout(multivariate_normal(mu0, S0), policy, H, plant, cost)

policy.fcn = lambda m, s: concat(congp, gaussian_sin, policy, m, s)

dynmodel = gpmodel()
dynmodel.fcn = dynmodel.gp0
train(dynmodel, plant, policy, x, y)

# Draw rollout.
L = 0.6
x0 = latent[:, 0]
y0 = np.zeros_like(x0)
x1 = x0 + L * sin(latent[:, 3])
y1 = -L * cos(latent[:, 3])

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax.set_aspect("equal")
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


def animate(i):
    linex = [x0[i], x1[i]]
    liney = [y0[i], y1[i]]
    line.set_data(linex, liney)
    time_text.set_text("time = %.1fs" % (i * dt))
    return line, time_text


ani = animation.FuncAnimation(
    fig, animate, np.arange(len(latent)), interval=100, blit=True)
plt.show()
