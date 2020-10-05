import pandas as pd
import numpy as np
from numpy.random import Generator, PCG64  # numpy randon number generator


def Psamp(X, r, rg):
    """Proposal """
    Y = 2 * r * rg.random() + (X - r)
    return Y


def Rsamp(v, X, r, rg):
    """Metropolic Hastings sampler"""
    Y = Psamp(X, r, rg)
    MH = np.exp((X * X - Y * Y) / (2 * v))
    if MH > 1:
        return Y
    U = rg.random()
    if U < MH:
        return Y
    return X


def MC_path(v, X0, r, q, N, rg):
    """Generate Monte Carlo path"""
    VX_arr = np.empty(N)
    for i in range(N):
        VX_arr[i] = X0
        X1 = Rsamp(v, X0, r, rg)
        X0 = X1

    B_hat = np.power(VX_arr, 2 * q).mean()
    return B_hat


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
"""
data = np.empty((len(N), 2))
for i in range(len(N)):
    sample_arr = np.empty(M)
    for p in range(M):
        sample_arr[p] = MC_path(v, 0, r, q, N[i], rg)
    sample_mean = sample_arr.mean()
    bias = np.abs(sample_mean - v)
    sample_var = sample_arr.var()
    data[i, 0] = bias
    data[i, 1] = sample_var

table = pd.DataFrame(data, columns=['bias', 'variance'], index= N)
"""

# we only change q to see the difference when dealing with harder ones
q = 2
data_q2 = np.empty((len(N), 2))
m2q = 3
for i in range(len(N)):
    sample_arr = np.empty(M)
    for p in range(M):
        sample_arr[p] = MC_path(v, 0, r, q, N[i], rg)
    sample_mean = sample_arr.mean()
    bias = np.abs(sample_mean - m2q)
    sample_var = sample_arr.var()
    data_q2[i, 0] = bias
    data_q2[i, 1] = sample_var

table_q2 = pd.DataFrame(data_q2, columns=['bias', 'variance'], index=N)

q = 4
data_q4 = np.empty((len(N), 2))
m2q = 105
for i in range(len(N)):
    sample_arr = np.empty(M)
    for p in range(M):
        sample_arr[p] = MC_path(v, 0, r, q, N[i], rg)
    sample_mean = sample_arr.mean()
    bias = np.abs(sample_mean - m2q)
    sample_var = sample_arr.var()
    data_q4[i, 0] = bias
    data_q4[i, 1] = sample_var

table_q4 = pd.DataFrame(data_q4, columns=['bias', 'variance'], index=N)