import numpy as np
from numpy import sin, cos, log
from numpy import ones, zeros, zeros_like, diag
from numpy.random import randn
from numpy.random import multivariate_normal as gaussian
import matplotlib.pyplot as plt
from matplotlib import animation

from base import rollout
from util import gaussian_trig

# z = [x1, v1, omiga, theta]
def dynamics(z, t, u):
	g = 9.82	# [m/s^2]
	l = 0.6		# [m]		length of pendulum
	m1 = 0.5	# [kg]		mass of pedulum
	m2 = 0.5	# [kg]		mass of cart
	b = 0.1		# [N/m/s]	coefficient of friction

	dzdt = zeros_like(z)

	dzdt[0] = z[1]
	dzdt[1] = (2*m1*l*(z[2]**2)*sin(z[3]) + 3*m1*g*sin(z[3])*cos(z[3])
			+ 4*u - 4*b*z[1]) / (4*(m1 + m2) - 3*m1*cos(z[3])**2)
	dzdt[2] = (-3*m1*l*(z[2]**2)*sin(z[3])*cos(z[3]) - 6*(m1 + m2)*g*sin(z[3])
			- 6*(u - b*z[1])*cos(z[3])) / (4*l*(m1 + m2) - 3*l*m1*cos(z[3])**2)
	dzdt[3] = z[2]

	return dzdt

odei = [0, 1, 2, 3]
dyno = [0, 1, 2, 3]
angi = [3]
dyni = [0, 1, 2, 4, 5]
poli = [0, 1, 2, 4, 5]
difi = [0, 1, 2, 3]

dt = 0.05
T = 5
H = int(T/dt)
mu0 = np.array([0, 0, 0, 0])
S0 = diag([0.01, 0.01, 0.01, 0.01])
N = 15
nc = 10

plant = {
	"dynamics":	dynamics,
	"noise":	diag(ones(4)*1e-4),
	"dt":		dt,
	"odei":		odei,
	"angi":		angi,
	"poli":		poli,
	"dyno":		dyno,
	"dyni":		dyni,
	"difi":		difi
}

policy = {"max_u": [10]}

m, s, c = gaussian_trig(mu0, S0, angi)
m = np.hstack([mu0, m])
c = np.dot(S0, c)
s = np.bmat([[S0, c], [c.T, s]]).A

p = {
	"inputs":	gaussian(m[poli], s[np.ix_(poli, poli)], nc),
	"targets":	.1*randn(nc, len(policy['max_u'])),
	"hyp":		log([1, 1, 1, 0.7, 0.7, 1, 0.01])
}
policy['p'] = p

x, y, latent = rollout(gaussian(mu0, S0), policy, H, plant, {})

# render rollout animation
l = 0.6
x1 = latent[:, 0]
y1 = np.zeros_like(x1)
x2 = x1 + l*sin(latent[:, 3])
y2 = -l*cos(latent[:, 3])

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax.set_aspect('equal')
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def animate(i):
	linex = [x1[i], x2[i]]
	liney = [y1[i], y2[i]]

	line.set_data(linex, liney)
	time_text.set_text('time = %.1fs' % (i*dt))
	return line, time_text

ani = animation.FuncAnimation(fig, animate, np.arange(len(y)),
	interval=30, blit=True)
plt.show()
