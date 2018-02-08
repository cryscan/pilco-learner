import numpy as np
from numpy import sin, cos, zeros, zeros_like
from scipy.integrate import odeint
from numpy.random import multivariate_normal as gaussian
import matplotlib.pyplot as plt
from matplotlib import animation

# z = [x1, v1, omiga, theta]
def dynamics(z, t, u):
	dzdt = zeros_like(z)

	dzdt[0] = z[1]
	dzdt[1] = (2*m1*l*(z[2]**2)*sin(z[3]) + 3*m1*g*sin(z[3])*cos(z[3])
			+ 4*u - 4*b*z[1]) / (4*(m1 + m2) - 3*m1*cos(z[3])**2)
	dzdt[2] = (-3*m1*l*(z[2]**2)*sin(z[3])*cos(z[3]) - 6*(m1 + m2)*g*sin(z[3])
			- 6*(u - b*z[1])*cos(z[3])) / (4*l*(m1 + m2) - 3*l*m1*cos(z[3])**2)
	dzdt[3] = z[2]

	return dzdt

dt = 0.05
T = 5
H = int(T/dt)			# total iterations

g = 9.82	# [m/s^2]
l = 0.6		# [m]		length of pendulum
m1 = 0.5	# [kg]		mass of pedulum
m2 = 0.5	# [kg]		mass of cart
b = 0.1		# [N/m/s]	coefficient of friction

maxu = 10	# [N]	maximum of force applied

# apply random actions
def policy(z):
	return np.random.uniform(-maxu, maxu)

# initial state
mu0 = np.array([0, 0, 0, 0])
S0 = np.diag([0.01, 0.01, 0.01, 0.01])

# rollout
y = np.atleast_2d(gaussian(mu0, S0))
for i in range(1, H):
	start= y[i-1]
	u = policy(start)
	next = odeint(dynamics, start, [0, dt], args=(u,))
	y = np.vstack((y, next[-1, :]))

# render rollout animation
x1 = y[:, 0]
y1 = np.zeros_like(x1)
x2 = x1 + l*sin(y[:, 3])
y2 = -l*cos(y[:, 3])

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
