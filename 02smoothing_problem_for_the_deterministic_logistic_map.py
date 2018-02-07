# smoothing problem for the deterministic logistic map

import numpy as np
import matplotlib.pyplot as plt

J = 1000                  # number of steps
r = 2                     # dynamics determined by r
gamma = 0.1               # observational noise variance is gamma^2
C0 = 0.5                 # prior initial condition variance
m0 = 0.7                  # prior initial condition mean
sd=1
np.random.seed(sd)        # choose random number seed

# set the truth
vt = np.zeros(J+1)
y = np.zeros(J)           # preallocate space to save time
vt[0] = 0.1               # truth initial condition
for j in range(J):
    # can be replaced by Psi for each problem
    vt[j+1] = r * vt[j] * (1-vt[j])             # create truth
    y[j] = vt[j+1] + gamma * np.random.randn()  #create data

# solution
v0 = np.arange(0.01,0.9905,0.0005) # construct vector of different initial data
Phidet = np.zeros(len(v0))
Idet = np.zeros(len(v0))
Jdet = np.zeros(len(v0))             # preallocate space
vv = np.zeros(J+1)                   # preallocate space 

# loop through initial conditions vv0, and compute log posterior I0(vv0) 
for j in range(len(v0)):
    vv[0] = v0[j] 
    Jdet[j] = 1. / 2. / C0 * (v0[j]-m0)**2 # background penalization
    for i in range(J):
        vv[i+1] = r * vv[i] * (1-vv[i])
        Phidet[j] = Phidet[j] + 1. / 2. / gamma**2 * (y[i]-vv[i+1])**2 # misfit functional
    Idet[j] = Phidet[j] + Jdet[j]

logconstant = np.trapz(x=v0,y=Idet) # calculate a log constant shift 
constant = np.trapz(x=v0,y=np.exp(logconstant-Idet))# calculate normalizing constant
P = np.exp(logconstant-Idet) / constant # normalize posterior distribution
prior = np.exp(-(v0-m0)**2/2./C0)/np.sqrt(2.*np.pi*C0) # calculate prior distribution

plt.figure(1,figsize=(12,5))

plt.plot(v0,prior,'k-',v0,P,'r--',linewidth=2)
plt.xlim(0, 1)
plt.xlabel('$v_0$')
plt.legend(('prior','J'),loc='upper center')
plt.savefig("02smoothing_problem_for_the_deterministic_logistic_map",dpi=600)
