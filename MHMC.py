"""
Code for Assignment 3, MonteCarlo methods, Oct 5, 2020
by Tao Li, taoli@nyu.edu

"""

import pandas as pd
import numpy as np
from numpy.random import Generator, PCG64  # numpy randon number generator


def Psamp(X, r, rg):
    """Proposal: a uniform distribution on [X-r, X+r]
    input: X: current MC sample, r: radius, rg: randomness generator
    """
    Y = 2 * r * rg.random() + (X - r)
    return Y


def Rsamp(v, X, r, rg):
    """Metropolic Hastings sampler with normal dist as the target
    input: v: variance of the normal dist, X: current mc sample, r: radius
    return: Y the next sample
    """
    # sample from the proposal
    Y = Psamp(X, r, rg)
    # metropolis hastings : since the proposal is uniform, we only need the target distribution
    MH = np.exp((X * X - Y * Y) / (2 * v))
    if MH > 1:
        return Y
    U = rg.random()
    if U < MH:
        return Y
    return X


def MC_path(v, X0, r, q, N, rg):
    """Generate Monte Carlo path
    input: v: variance of the normal dist, X0: the initial mc sample, r: radius, q: the power, N: length of mc path
    return an estimate of the expection of V(X), i.e., EX^{2q}
    """
    # allocate a np array for recording
    VX_arr = np.empty(N)
    for i in range(N):
        VX_arr[i] = X0
        # MH sampler for one move
        X1 = Rsamp(v, X0, r, rg)
        X0 = X1
    # sample average
    B_hat = np.power(VX_arr, 2 * q).mean()
    return B_hat


def moment_est(v, m, X0, r, q, N, M, rg):
    """Estimate the moment 2q
    input: v: variance of the normal dist, m: the precise moment for computing the bias X0: the initial mc sample,
    r: radius, q: the power, N: length of mc path, M: number of independent paths
    return: a table containing bias and variance, showing the convergence of B_hat
    """
    # allocate a np array for storing bias and variance
    data = np.empty((len(N), 2))
    for i in range(len(N)):
        # allocate a sample array for M independent paths
        sample_arr = np.empty(M)
        for p in range(M):
            # generate a mc path with length N[i]
            sample_arr[p] = MC_path(v, X0, r, q, N[i], rg)
        # compute the bias
        sample_mean = sample_arr.mean()
        bias = np.abs(sample_mean - m)
        # compute the variance
        sample_var = sample_arr.var()
        data[i, 0] = bias
        data[i, 1] = sample_var
    table = pd.DataFrame(data, columns=['bias', 'variance'], index=N)
    return table


# main program
seed = 12344  # change this to get different answers
bg = PCG64(seed)  # instantiate a bit generator
rg = Generator(bg)  # instantiate a random number generator with this bit generator

# parameters for showing the convergence
r = 1
v = 1
q = 1
M = 1000
N = [100, 1000, 5000, 10000, 20000]
X0 = 0

# q=1 then the second moment is just the variance v, i.e, m=v. The following code for showing the convergence is the
# same as the function moment_est, hence we omit comments

data = np.empty((len(N), 2))
for i in range(len(N)):
    sample_arr = np.empty(M)
    for p in range(M):
        sample_arr[p] = MC_path(v, X0, r, q, N[i], rg)
    sample_mean = sample_arr.mean()
    bias = np.abs(sample_mean - v)
    sample_var = sample_arr.var()
    data[i, 0] = bias
    data[i, 1] = sample_var
table_q1 = pd.DataFrame(data, columns=['bias', 'variance'], index=N)

# we now estimate the auto-correlation time tau. We use the mc path where N = 20000
# the sample variance is given by sample_var (the last iterate), whereas the true one var(V(X)) is 2v^2
true_var = 2 * v * v
tau = sample_var * N[-1] / true_var

# we only change q to see the difference when dealing with harder ones
N = [5000, 10000, 20000, 50000]
# when q=2, i.e., 4th moment = 3

table_q2 = moment_est(v, 3, X0, r, 2, N, M, rg)
# when q=4, i.e., 8th moment = 105

table_q4 = moment_est(v, 105, X0, r, 4, N, M, rg)

# when q = 6 , the 12th moment = 10395

table_q6 = moment_est(v, 10395, X0, r, 6, N, M, rg)

# We finally experiment with different r
q = 1
N = [1000, 5000, 10000]
# r= 0.1
table_r01 = moment_est(v, v, X0, 0.1, q, N, M, rg)

# r= 0.5
table_r05 = moment_est(v, v, X0, 0.5, q, N, M, rg)

# r= 10
table_r5 = moment_est(v, v, X0, 10, q, N, M, rg)
