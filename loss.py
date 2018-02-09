import numpy as np
from numpy import eye, zeros, exp, sqrt
from numpy import dot, linalg

def loss_sat(cost, m, s, return_derivatives=False):
	"""
	Compute expectation and variance of a saturating cost
	1 - exp(-(x - z)'W(x - z)/2)
	and their derivatives, where x ~ N(m, s), z is a target state,
	and W is a weighting matrix.
	"""
	D = len(m)

	W = eye(D) if cost.get('W') is None else cost['W']
	z = zeros(D) if cost.get('z') is None else cost['z']

	SW = dot(s, W)
	iSpW = linalg.solve((eye(D) + SW).T, W.T)

	L = -exp(dot(dot(-(m - z), iSpW), (m - z).T)/2) \
		/ sqrt(linalg.det(eye(D) + SW))

	if not return_derivatives:
		return L
