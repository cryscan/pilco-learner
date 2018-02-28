import math

import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd.numpy import sin, cos, log
from autograd.numpy.random import randn, multivariate_normal
from matplotlib import animation

from pilco import Empty
from pilco.base import rollout, train, propagate, learn
from pilco.control import congp, concat
from pilco.gp import GPModel
from pilco.loss import Loss
from pilco.util import gaussian_trig, gaussian_sin, fill_mat


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


def loss_cp(self, m, s):
    D0 = np.size(s, 1)
    D1 = D0 + 2 * len(self.angle)
    M = m
    S = s

    ell = self.p
    Q = np.dot(np.vstack([1, ell]), np.array([[1, ell]]))
    Q = fill_mat(Q, np.zeros((D1, D1)), [0, D0], [0, D0])
    Q = fill_mat(ell**2, Q, [D0 + 1], [D0 + 1])

    target = gaussian_trig(self.target, 0 * s, self.angle)[0]
    target = np.hstack([self.target, target])
    i = np.arange(D0)
    m, s, c = gaussian_trig(M, S, self.angle)
    q = np.dot(S[np.ix_(i, i)], c)
    M = np.hstack([M, m])
    S = np.vstack([np.hstack([S, q]), np.hstack([q.T, s])])

    w = self.width if hasattr(self, "width") else [1]
    L = np.array([0])
    S2 = np.array(0)
    for i in range(len(w)):
        self.z = target
        self.W = Q / w[i]**2
        r, s2, c = self.loss_sat(M, S)
        L = L + r
        S2 = S2 + s2

    return L / len(w)


def draw_rollout(latent):
    x0 = latent[:, 0]
    y0 = np.zeros_like(x0)
    x1 = x0 + 0.6 * sin(latent[:, 3])
    y1 = -0.6 * cos(latent[:, 3])

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
        trail = math.floor(i / (H + 1))
        time_text.set_text("trail %d, time = %.1fs" % (trail, i * dt))
        return line, time_text

    interval = math.ceil(T / dt)
    ani = animation.FuncAnimation(
        fig, animate, np.arange(len(latent)), interval=interval, blit=True)
    ani.save('cart_pole.mp4', fps=20)
    plt.show()


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
N = 6
nc = 10

plant = Empty()
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

policy = GPModel()
policy.max_u = [10]
policy.p = {
    'inputs': multivariate_normal(m[poli], s[np.ix_(poli, poli)], nc),
    'targets': 0.1 * randn(nc, len(policy.max_u)),
    'hyp': log([1, 1, 1, 0.7, 0.7, 1, 0.01])
}

Loss.fcn = loss_cp
cost = Loss()
cost.p = 0.5
cost.gamma = 1
cost.width = [0.25]
cost.angle = plant.angi
cost.target = np.array([0, 0, 0, np.pi])

start = multivariate_normal(mu0, S0)
x, y, L, latent = rollout(start, policy, plant, cost, H)
policy.fcn = lambda m, s: concat(congp, gaussian_sin, policy, m, s)

for i in range(N):
    dynmodel = GPModel()
    dynmodel.fcn = dynmodel.gp0
    train(dynmodel, plant, policy, x, y)
    result = learn(mu0, S0, dynmodel, policy, plant, cost, H)

    start = multivariate_normal(mu0, S0)
    x_, y_, L, latent_ = rollout(start, policy, plant, cost, H)
    x = np.vstack([x, x_])
    y = np.vstack([y, y_])
    latent = np.vstack([latent, latent_])
    print("Test loss: %s", np.sum(L))

save_file = open('cart_pole.json', 'w')
save_file.write(str(result))
save_file.close()
draw_rollout(latent)
