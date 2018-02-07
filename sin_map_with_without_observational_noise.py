# behaviour of sin map with and without observational noise

import numpy as np
import matplotlib.pyplot as plt

J = 10000              # number of steps
alpha = 2.5            # dynamics determined by alpha
sigma = 0.25           # dynamics noise variance is sigma^2
sd = 1
np.random.seed(sd)     # choose random number seed
v = np.zeros(J+1)
vnoise = np.zeros(J+1) # preallocate space
v[0] = 1
vnoise[0] = 1          # initial conditions

for ii in range(J):
    v[ii+1] = alpha * np.sin(v[ii])
    vnoise[ii+1] = alpha * np.sin(vnoise[ii]) + sigma * np.random.randn()

plt.figure(1,figsize=(12,5))

plt.subplot(121)
plt.plot(range(J+1),v)
plt.title(r'Deterministic dynamics', fontsize=15)
plt.xlim(0, 100)
plt.xlabel('$j$')
plt.ylabel('$u_j$')

plt.subplot(122)
plt.plot(range(J+1),vnoise)
plt.title(r'Stochastic dynamics', fontsize=15)
plt.xlim(0, 10000)
plt.xlabel('$j$')
plt.ylabel('$u_j$')

plt.savefig("01sin_map_with_without_observational_noise",dpi=600)
