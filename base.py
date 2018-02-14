import numpy as np
from numpy import zeros, hstack, size, diag
from numpy.random import rand, randn
from numpy.random import multivariate_normal as gaussian
from scipy.integrate import odeint

from util import gaussian_trig

def rollout(start, policy, H, plant, cost, return_cost=False):
	"""
	Generate a state trajectory using an ODE solver.

	Input arguments:
	start		initial states	(nX,)
	policy		policy structure
	.fcn		policy function (use random actions if empty)
	.p		parameters
	.max_u		control input saturation values	(nU,)
	H		rollout horizon in steps
	plant		dynamical system structure
	.poli		indices for states passed to policy
	.dyno		indices for states passed to cost
	.odei		indices for states passed to the ode solver
	.angi		indices of angles in states
	cost		cost structure

	Return:
	x		matrix of observed states	(H, nX + nU)
	y		matrix of observed successor states	(H, nX)
	L		cost incurred at each time step	(H,)
	"""
	odei = plant['odei']
	poli = plant['poli']
	dyno = plant['dyno']
	angi = plant['angi']

	nX = len(odei)
	nU = len(policy['max_u'])
	nA = len(angi)

	# initializations
	state = start
	x = zeros([H + 1, nX + 2*nA])
	x[0, odei] = gaussian(start, plant['noise'])
	
	u = zeros([H, nU])
	y = zeros([H, nX])
	L = zeros(H)
	latent = zeros([H + 1, size(state) + nU])

	for i in range(H):
		s = x[i, dyno]
		a, _, _ = gaussian_trig(s, diag(zeros(size(s))), angi)
		s = hstack([s, a])
		x[i, -2*nA:] = s[-2*nA:]

		f = policy.get('fcn')
		if f is None:	# perform random actions
			u[i, :] = policy['max_u']*(2*rand(nU) - 1)
		else:	# or apply policy
			u[i, :] = f(policy, s[poli], zeros(len(poli)))
		latent[i, :] = hstack([state, u[i, :]])
		
		# solve dynamics
		dynamics = plant['dynamics']
		dt = plant['dt']
		next = odeint(dynamics, state[odei], [0, dt], args=(u[i, :],))
		state = next[-1, :]
		x[i + 1, odei] = gaussian(state[odei], plant['noise'])

		if return_cost:
			L[i] = cost['fcn'](cost, state[dyno], zeros(len(dyno)))
		
	y = x[1:H + 1, 0:nX]
	x = hstack([x[0:H, :], u[0:H, :]])
	latent[H, 0:nX] = state

	if return_cost: return x, y, L, latent
	return x, y, latent
