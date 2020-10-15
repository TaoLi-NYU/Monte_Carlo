"""
Code for Assignment 4, MonteCarlo methods, Oct 15, 2020
by Tao Li, taoli@nyu.edu

"""
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import Generator, PCG64  # numpy randon number generator
from scipy.stats import multivariate_normal

def rho(x):
    """
    The pdf of target distribution
    """
    y = np.zeros(len(x))
    # x-y gives \sum_{j}x_{j+1}-x_j, so we shift x by one index
    y[:-1] = x[1:]
    diff_square = np.square(x-y).sum()
    prob = np.exp(-(diff_square + x[0]*x[0]) / 2)
    return prob

def prop(y,r,rg):
    """
    Proposal
    input: y: current sample, r: proposal size std dev
    return: another sample from N(y, r^2 I)
    """
    mean = y
    cov = r*r*np.eye(len(y))
    x = rg.multivariate_normal(mean, cov)
    return x

def rsample(y,r,rg):
    """
    resampling
    input: y: current sample, r: proposal size
    """
    # have a proposal
    x = prop(y,r,rg)
    # evaluate the pdf of target
    rhoX = rho(x)
    rhoY = rho(y)
    cov = r*r*np.eye(len(y))
    # evalute the pdf of the proposal
    PYX = multivariate_normal(mean= x, cov = cov).pdf(y)
    PXY = multivariate_normal(mean= y, cov = cov).pdf(x)
    # metropolis
    MH = (rhoX*PYX)/(rhoY*PXY)
    if MH > 1:
        return  x
    U = rg.random()
    if U<MH:
        return x
    return y

def MH_global(x0, r,N, rg):
    """
    Global metropolis
    input: x0: initial sample, r: proposal size, N: length
    return: sample path
    """
    # allocation
    path = np.empty((N,len(x0)))
    path[0, :] = x0
    # total: the number of all samples generated
    # counter: the number of valid samplers
    total = 1
    counter = 0
    while counter < N-1:
        total += 1
        # MH sampler for one move
        x1 = rsample(x0, r,rg)
        # if x1 is non negative then
        if x1.min() >= 0:
            counter += 1
            path[counter,:] = x1
        x0 = x1
    ratio = N/total
    return ratio, path

## partial sampling Gibbs
def resamp(i,x,rg):
    """
    Resampling for i component using Gibbs
    input: i: the index of the compoenent, x: current sample
    return a new sample
    """
    # we explain the Gibbs in the hw, so we omit the comments here
    if i ==0:
        y = x[1]/2+np.sqrt(1/2)*rg.normal()
    elif i == len(x)-1:
        y = x[-2]/2 + np.sqrt(1/2)*rg.normal()
    else:
        y = (x[i-1]+x[i+1])/2+np.sqrt(1/2)*rg.normal()

    return y



def partial_samp(x,rg):
    """
    partial sampling
    input: x: the current sample
    return a new sample after a sweep
    """
    for i in range(n):
        x[i] = resamp(i,x,rg)
    return x

def MC_sweep(x0,N,rg):
    """
    MC using partial sampling
    input: x0: initial sample, N: length
    return: a sample path
    """
    # allocation
    path = np.empty((N, len(x0)))
    counter = 0
    path[0,:] = x0
    while counter < N-1:
        x1 = partial_samp(x0,rg)
        # if x1 is non-negative then it is valid, keep x1
        if x1.min()>= 0:
            counter += 1
            path[counter,:] = x1
        x0 = x1
    return path

#
def autocov_est(v_arr,t):
    """
    Estimate the auto-covariance
    input
    """
    length = len(v_arr)
    if t>length-1:
        print('invalid t')
        return
    v_bar = v_arr.mean()
    v_diff0 = v_arr[0:N-1-t]-v_bar
    v_difft = v_arr[t:N-1]- v_bar
    est = np.multiply(v_diff0,v_difft).mean()
    return est


# main program
seed = 12344  # change this to get different answers
bg = PCG64(seed)  # instantiate a bit generator
rg = Generator(bg)  # instantiate a random number generator with this bit generator

# we first show that samplers are correct by plotting the histogram
# 3-d vecotr and run for 5000 steps with proposal size 1
n = 3
N = 5000
x0 = np.ones(n)
r= 1
ratio,path_g = MH_global(x0,r, N, rg)

path_s = MC_sweep(x0,N,rg)
# compute the V function
V_g = np.mean(path_g,axis= 1)
V_s = np.mean(path_s,axis = 1)

plt.hist(V_g, bins='auto',alpha=0.5, label='global')
plt.hist(V_s, bins='auto', alpha=0.5, label='sweep')
plt.legend(loc='upper right')
plt.show()

#Experiment auto-covariance with a list of n values
# we set t=800
T = 800
n_list = [3, 5, 10, 15, 20 ]
# allocation
autocov_g = np.zeros((len(n_list),T))
autocov_s = np.zeros((len(n_list),T))
# for various combinations of r and n
for i in range(len(n_list)):
    # initialization : a new x0
    x0 = np.ones(n_list[i])
    _,path_g = MH_global(x0, r, N, rg)
    path_s = MC_sweep(x0, N, rg)
    V_g = np.mean(path_g, axis=1)
    V_s = np.mean(path_s, axis=1)
    # estimate the auto cov
    for t in range(T):
        autocov_g[i,t] = autocov_est(V_g,t)
        autocov_s[i,t] = autocov_est(V_s,t)

plt.plot(np.linspace(0,T,T), autocov_g.T)
plt.legend(('n=3','n=5','n=10','n=15','n=20'),loc='upper right')
plt.xlabel("time t")
plt.ylabel("auto covariance for global")
plt.show()

plt.plot(np.linspace(0,T,T), autocov_s.T)
plt.legend(('n=3','n=5','n=10','n=15','n=20'),loc='upper right')
plt.xlabel("time t")
plt.ylabel("auto covariance for sweep")
plt.show()

#Experiment proposal size
N=5000
# various n values and r values
n_list = [3, 5, 10, 15]
r_list = [0.5, 1, 3, 5]
# allocation
ratio = np.empty((len(r_list),len(n_list)))
# for different combiantions of r and n
for i in range(len(n_list)):
    x0 = np.ones(n_list[i])
    for j in range(len(r_list)):
        ratio[j,i], _ = MH_global(x0, r_list[j], N, rg)

plt.plot(r_list, ratio,'o-')
plt.legend(('n=3','n=5','n=10','n=15'),loc='upper right')
plt.xlabel("square root of variance")
plt.ylabel("sampling efficiency")
plt.show()
