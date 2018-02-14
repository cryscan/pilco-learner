
# coding: utf-8

# In[1]:


import autograd.numpy as np
from autograd.numpy import sin, cos, exp


# In[2]:


def gaussian_trig(m, v, i):
    d, I = len(m), len(i)
    mi = m[i]
    vi = v[np.ix_(i, i)]
    vii = np.diag(vi)
    
    M = np.vstack([exp(-vii/2) * sin(mi),
                   exp(-vii/2) * cos(mi)])
    M = M.flatten(order='F')
    
    mi = mi.reshape(I, 1)
    vii = vii.reshape(I, 1)
    lq = -(vii + vii.T)/2
    q = exp(lq)
    
    U1 = (exp(lq + vi) - q) * sin(mi - mi.T)
    U2 = (exp(lq - vi) - q) * sin(mi + mi.T)
    U3 = (exp(lq + vi) - q) * cos(mi - mi.T)
    U4 = (exp(lq - vi) - q) * cos(mi + mi.T)
    
    V = np.vstack([np.hstack([U3 - U4, U1 + U2]),
                  np.hstack([(U1 + U2).T, U3 + U4])]) / 2
    V = np.vstack([np.hstack([V[::2, ::2], V[::2, 1::2]]),
                  np.hstack([V[1::2, ::2], V[1::2, 1::2]])])
    
    C = np.hstack([np.diag(M[1::2]), -np.diag(M[::2])])
    C = np.hstack([C[:, ::2], C[:, 1::2]])
    T = np.zeros([d, I])
    for k, l in enumerate(i): T[l, k] = 1
    C = T @ C
    
    return M, V, C


# In[3]:


def gaussian_sin(m, v, i):
    d, I = len(m), len(i)
    mi = m[i]
    vi = v[np.ix_(i, i)]
    vii = np.diag(vi)
    M = exp(-vii/2) * sin(mi)
    
    mi = mi.reshape(I, 1)
    vii = vii.reshape(I, 1)
    lq = -(vii + vii.T)/2
    q = exp(lq)
    V = ((exp(lq + vi) - q) * cos(mi - mi.T)
         - (exp(lq - vi) - q) * cos(mi + mi.T)) / 2
    
    C = np.diag((exp(-vii/2) * cos(mi)).flatten())
    T = np.zeros([d, I])
    for k, l in enumerate(i): T[l, k] = 1
    C = T @ C
    
    return M, V, C

