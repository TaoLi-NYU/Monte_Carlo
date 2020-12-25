"""
Code for Assignment 10, MonteCarlo methods, Dec 23, 2020
by Tao Li, taoli@nyu.edu

"""
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import Generator, PCG64  # numpy randon number generator

def resamp(X,s,rg):
    """Gibbs sampler for resampling X
    input: X: current sample, s: threshold
    """

    for i in range(len(X)):
        y = rg.random()
        if np.sum(X)+y-X[i]>s:
            X[i] = y
    return X

def MC_path(N,X,s,rg):
    """MCMC for the target distribution rho_s, with N being the sample length
    """
    X0 = X
    path = np.zeros((len(X0),N))
    for i in range(N):
        path[:,i] = X0
        X0 = resamp(X0,s,rg)
    return path

def est_s1(n,m,rg):
    """
    Estimation of s1 by m independent samples
    input: n: dimension, m: num of samples
    output: s1 and a sample
    """
    samples = rg.random(size=(n,m))
    sum_samp = np.sum(samples,axis=0)
    sum_arg = np.argsort(sum_samp)
    mid = sum_arg[int(m/2)-1]
    s1 = sum_samp[mid]
    X = samples[:,mid]
    return s1, X


def est_sr(N,X,s,rg):
    """
    Estimation of the next s by histogram bifurcation applied to samples from current s
    input: N: sample length
    """
    path_s = MC_path(N,X,s,rg)
    S_arr = np.sum(path_s,axis=0)
    S_arg = np.argsort(S_arr)
    mid = S_arg[int(N/2)-1]
    sr = S_arr[mid]
    X = path_s[:,mid]
    return sr, X

def sr_sample(n,r,rg, m=1000, N=2000):
    """
    Find sr for a given r
    input: n: dimension, r: given r, m: # initial samples for estimating s1,
    N:sample length of the generated MC path
    """
    s1 ,X = est_s1(n,m,rg)
    for i in range(r):
        sr, Xr = est_sr(N,X,s1,rg)
        X = Xr
        s1 =sr
    return sr, Xr

def MC_Spath(N,X,s,rg):
    """
    Generate MC sample path for variable S
    N: sample lenght, X: an inital sample, s:condition on S>s
    """
    X0 = X
    path = np.zeros((len(X0), N))
    for i in range(N):
        path[:,i] = X0
        X0 = resamp(X0, s, rg)
    Spath = np.sum(path, axis=0)
    return  Spath

def autocov_est(v_arr,t):
    """
    Estimate the auto-covariance
    input: v_arr value function here it is the sample path
    """
    N = len(v_arr)
    if t>N-1:
        print('invalid t')
        return
    v_bar = v_arr.mean()
    v_diff0 = v_arr[0:N-1-t]-v_bar
    v_difft = v_arr[t:N-1]- v_bar
    est = np.multiply(v_diff0,v_difft).mean()
    return est

def auto_corr(v_arr, T):
    """
    Estimate the auto-correlation
    input: v_arr: sample path, T: maximum time
    return: auto-correlation estimation
    """
    autocorr_est = np.zeros(T)
    for t in range(T):
        autocorr_est[t] = autocov_est(v_arr, t)
    autocorr_est = autocorr_est / autocorr_est[0]
    return autocorr_est

def auto_time(autocorr, w):
    """
    Estimate the auto-correlation time by self-consistency
    input: autocorr: auto-correlation estimation, w: window size
    reeturn: estiamte of tau
    """
    T = len(autocorr)
    tau = 1
    t = 0
    while True:
        if t > w * tau:
            break
        t = t + 1
        if t == T:
            print("the run was too short")
            return
        tau = tau + 2 * autocorr[t]

    return tau

def experiment(r,n,rg,plot=True, T=2000, m=2000, N=5000):
    """
    We experiment with different r and n
    """
    sr, Xr = sr_sample(n=n, r=r, rg=rg, m=2000)
    Spath = MC_Spath(N=5000, X=Xr, s=sr, rg=rg)
    auto_corr_s = auto_corr(Spath, 2000)
    tau = auto_time(auto_corr_s, 5)
    if plot :
        plt.plot(np.linspace(0, T, T), auto_corr_s)
        plt.xlabel("time t")
        plt.ylabel("auto correlation")
        title_text = "auto correlation function under r={r:d} , n={n:d} "
        plt.title(title_text.format(r=r, n=n))
        plt.show()
    return tau



# main
seed = 12344  # change this to get different answers
bg = PCG64(seed)  # instantiate a bit generator
rg = Generator(bg)  # instantiate a random number generator with this bit generator

# r stands for the subscript of s_r,  try r= 10 ,15, 20

# n is the dimension of X

n_set = [10,15,20,25,30]
r_set = [10,15,20,25,30]
tau_rec = np.zeros((5,5))
for i in range(5):
    for j in range(5):
        tau_rec[i,j] = experiment(r=r_set[i], n=n_set[j],rg=rg,plot=False)



