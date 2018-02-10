# MCMC pCN Dynamics algorithm for sin map with noise

import numpy as np
import matplotlib.pyplot as plt

J = 10                                           # number of steps
alpha = 2.5                                      # dynamics determined by alpha
gamma = 1                                        # observational noise variance is gamma^2
sigma = 1                                        # dynamics noise variance is sigma^2
C0 = 1                                           # prior initial condition variance
m0 = 0                                           # prior initial condition mean
sd = 0
np.random.seed(sd)                               # choose random number seed

# truth
vt = np.zeros(J+1)
vt[0] = m0 + np.sqrt(C0) * np.random.randn()     # truth initial condition 
ut = np.zeros(J+1)
ut[0] = vt[0]
y = np.zeros(J) 
Phi = 0
for j in range(J):
    ut[j+1] = sigma * np.random.randn()
    vt[j+1] = alpha * np.sin(vt[j]) + ut[j+1]    # create truth
    y[j] = vt[j+1] + gamma * np.random.randn()   # create data
    # calculate log likelihood phi(u;y) 
    Phi = Phi + 1. / 2. / gamma**2 * (y[j]-vt[j+1])**2

# solution
# Markov Chain Monte Carlo: N forward steps
N = 100000                                       # number of samples
beta = 0.2                                       # step-size of pCN walker 
u = ut
v = vt                                           # truth initial condition (or update Phi)
V = np.zeros([N,J+1]) 
n = 1 
bb = 0 
rat = np.zeros(N)
rat[0] = 0
m = np.zeros(J+1)
m[0] = m0
vv = np.zeros(J+1)
while n < N: 
    iota = sigma * np.random.randn(J+1)    # Gaussian prior sample 
    iota[0] = np.sqrt(C0)*np.random.randn()           
    w = m + np.sqrt(1-beta**2) * (u-m) + beta * iota  # propose sample from the pCN walker 
    vv[0] = w[0]
    Phiprop = 0    
    for j in range(J):
        vv[j+1] = alpha * np.sin(vv[j]) + w[j+1]      # create path
        Phiprop = Phiprop + 1. / 2. / gamma**2 * (y[j]-vv[j+1])**2
    
    if np.random.rand() < np.exp(Phi-Phiprop):         # accept or reject proposed sample
        u = w
        v = vv
        Phi = Phiprop
        bb = bb+1                   # update the Markov chain
    rat[n] = bb / n                 # running rate of acceptance
    V[n,:] = v                      # store the chain
    n = n + 1

# plot acceptance ratio and cumulative sample mean
plt.figure(1,figsize=(12,5))

plt.subplot(121)
plt.plot(range(N),rat)
plt.xlim(0, N)

plt.subplot(122)
plt.plot(range(N),np.cumsum(V[0:N,0])/range(N))
plt.xlim(0, N)
plt.xlabel('samples N')
plt.ylabel('$(1/N) \Sigma_{n=1}^N v_0^{(n)}$')

plt.savefig("06MCMC_pCN_Dynamics_algorithm_for_sin_map_with_noise",dpi=600)

