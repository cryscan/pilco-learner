import numpy as np
from numpy import sin, cos, exp
from numpy import arange, diag, zeros, ones
from numpy import vstack, hstack

def trig(m, v, i, return_derivatives=False):
	"""
	Compute moments of the saturating function.

	Input arguments:
	m		mean vector	(d,)
	v		covariance matrix	(d, d)
	i		vector od indices	(I,)
	"""
	d, I = len(m), len(i)
	mi = m[i]
	vi = v[np.ix_(i, i)]
	vii = diag(vi)

	M = vstack((exp(-vii/2)*sin(mi), exp(-vii/2)*cos(mi))).flatten(order='F')

	mi = mi.reshape(I, 1)
	vii = vii.reshape(I, 1)
	lq = -(vii - vii.T)/2
	q = exp(lq)

	U1 = (exp(lq + vi) - q)*sin(mi - mi.T)
	U2 = (exp(lq - vi) - q)*sin(mi + mi.T)
	U3 = (exp(lq + vi) - q)*cos(mi - mi.T)
	U4 = (exp(lq - vi) - q)*cos(mi + mi.T)

	V = zeros((2*I, 2*I))
	V[::2, ::2] = U3 - U4
	V[1::2, 1::2] = U3 + U4
	V[::2, 1::2] = U1 + U2
	V[1::2, ::2] = V[::2, 1::2].T

	C = zeros((d, 2*I))
	C[i, ::2] = diag(M[1::2])
	C[i, 1::2] = diag(M[::2])

	if not return_derivatives:
		return M, V, C

	dVdm = zeros((2*I, 2*I, d))
	dCdm = zeros((d, 2*I, d))
	dVdv = zeros((2*I, 2*I, d, d))
	dCdv = zeros((d, 2*I, d, d))
	dMdm = C.T

	for j in range(I):
		u = zeros((I, 1))
		u[j] = 0.5
		sj, cj, ij = 2*j, 2*j + 1, i[j]

		dVdm[::2, ::2, ij] = -U1*(u - u.T) + U2*(u + u.T)
		dVdm[1::2, 1::2, ij] = -U1*(u - u.T) - U2*(u + u.T)
		dVdm[::2, 1::2, ij] = U3*(u - u.T) + U4*(u + u.T)
		dVdm[1::2, ::2, ij] = dVdm[::2, 1::2, ij].T

		dVdv[sj, sj, ij, ij] = exp(-vii[j])/2 \
			* (1 + (2*exp(-vii[j]) - 1)*cos(2*mi[j]))
		dVdv[cj, cj, ij, ij] = exp(-vii[j])/2 \
			* (1 - (2*exp(-vii[j]) - 1)*cos(2*mi[j]))
		dVdv[sj, cj, ij, ij] = exp(-vii[j])/2 \
			* (1 - 2*exp(-vii[j])*sin(2*mi[j]))
		dVdv[cj, sj, ij, ij] = dVdv[sj, cj, ij, ij]

		for k in range(I):
			if k == j: continue
			sk, ck, ik = 2*k, 2*k + 1, i[k]

			dVdv[sj, sk, ij, ik] = 0.5 \
				* (exp(lq[j, k] + vi[j, k]) * cos(mi[j] - mi[k]) \
				+ exp(lq[j, k] - vi[j, k]) * cos(mi[j] + mi[k]))
			dVdv[sj, sk, ij, ij] = -V[sj, sk]/2
			dVdv[sj, sk, ik, ik] = -V[sj, sk]/2
			
			dVdv[cj, ck, ij, ik] = 0.5 \
				* (exp(lq[j, k] + vi[j, k]) * cos(mi[j] - mi[k]) \
				- exp(lq[j, k] - vi[j, k]) * cos(mi[j] + mi[k]))
			dVdv[cj, ck, ij, ij] = -V[cj, ck]/2
			dVdv[cj, ck, ik, ik] = -V[cj, ck]/2

			dVdv[cj, sk, ij, ik] = -0.5 \
				* (exp(lq[j, k] + vi[j, k]) * sin(mi[j] - mi[k]) \
				+ exp(lq[j, k] - vi[j, k]) * sin(mi[j] + mi[k]))
			dVdv[cj, sk, ij, ij] = -V[cj, sk]/2
			dVdv[cj, sk, ik, ik] = -V[cj, sk]/2

			dVdv[sj, ck, ij, ik] = 0.5 \
				* (exp(lq[j, k] + vi[j, k]) * sin(mi[j] - mi[k]) \
				- exp(lq[j, k] - vi[j, k]) * sin(mi[j] + mi[k]))
			dVdv[sj, ck, ij, ij] = -V[sj, ck]/2
			dVdv[sj, ck, ik, ik] = -V[sj, ck]/2

		dCdm[ij, sj, ij] = -M[sj]
		dCdm[ij, cj, ij] = -M[cj]
		dCdv[ij, sj, ij, ij] = -C[ij, sj]/2
		dCdv[ij, cj, ij, ij] = -C[ij, cj]/2

	dMdv = dCdm.transpose((1, 0, 2))/2

	return M, V, C, dMdm, dVdm, dCdm, dMdv, dVdv, dCdv
