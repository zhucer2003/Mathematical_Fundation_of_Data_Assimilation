# MCMC RWM algorithm for logistic map

import numpy as np
import matplotlib.pyplot as plt

J = 5                                   # number of steps
r = 4                                   # dynamics determined by alpha
gamma = 0.2                             # observational noise variance is gamma^2
C0 = 0.01                               # prior initial condition variance
m0 = 0.5                                # prior initial condition mean
sd = 10                                 
np.random.seed(sd)                      # choose random number seed

# set the truth
vt = 0.3
vv = np.zeros(J+1)
vv[0] = vt                              # truth initial condition
Jdet = 1. / 2 / C0 * (vt-m0)**2         # background penalization
Phidet = 0  
y = np.zeros(J)                            # initialization model-data misfit functional
for j in range(J):
    # can be replaced by Psi for each problem
    vv[j+1] = r * vv[j] * (1-vv[j])     # create truth
    y[j] = vv[j+1] + gamma * np.random.randn() # create data
    Phidet = Phidet + 1. / 2 / gamma**2 * (y[j]-vv[j+1])**2 # misfit functional

Idet = Jdet + Phidet                    # compute log posterior of the truth

# solution
# Markov Chain Monte Carlo: N forward steps of the Markov Chain on R (with truth initial condition)
N = 100000                              # number of samples
V = np.zeros(N)                         # preallocate space to save time
beta = 0.05                             # step-size of random walker 
v = vt                                  # truth initial condition (or else update I0) 
n = 1 
bb = 0 
rat = np.zeros(N)
rat[0] = 0 

while n < N:
    w = v + np.sqrt(2*beta) * np.random.randn() # propose sample from random walker
    vv[0] = w
    Jdetprop = 1. / 2. / C0 * (w-m0)**2  # background penalization
    Phidetprop = 0	
    for i in range(J):
    	vv[i+1] = r * vv[i] * (1-vv[i])
    	Phidetprop = Phidetprop + 1. / 2. / gamma**2 * (y[i]-vv[i+1])**2
    Idetprop = Jdetprop + Phidetprop     # compute log posterior of the proposal 	

    if np.random.rand() < np.exp(Idet-Idetprop):      # accept or reject proposed sample
        v = w
        Idet = Idetprop 
        bb = bb+1           # update the Markov chain
    rat[n] = bb / n         # running rate of acceptance
    V[n] = v                # store the chain
    n = n + 1 

dx = 0.0005 
v0 = np.arange(0.01,0.99+dx,dx)
Z, _ = np.histogram(V, bins=v0)       # construct the posterior histogram 

plt.figure(1,figsize=(12,5))
plt.plot(v0[1:],Z/np.trapz(x=v0[1:],y=Z),'k-',linewidth=2) 
plt.xlim(0, 1)
plt.savefig("03MCMC_RWM_algorithm_for_logistic_map",dpi=600)
