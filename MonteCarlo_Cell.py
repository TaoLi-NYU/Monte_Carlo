"""
Code for Assignment 1, MonteCarlo methods, sep 7, 2020
by Tao Li, taoli@nyu.edu

"""

from numpy.random import Generator, PCG64  # numpy randon number generator
import numpy as np
import time


def mc_cellfire(F, Lambda, r, rg, X0=0):
    """
    generate a sample path starting from X0=0 until the cell fires
    F: fire threshold
    Lambda: arrival rate
    r: leaking rate
    rg: a random number generator, already instantiated and seeded
    reutn the first firing time tau and n_tau the poisson arrivals up to tau
    """
    tau = 0  # a timer
    n_tau = 0  # a counter
    x_0 = X0  # current time cell energy
    x_1 = X0  # next arrival time cell energy
    while x_1 < F:  # if
        u = rg.random()  # random number, uniform distribution
        t = -1 / Lambda * np.log(u)  # poison arrival
        x_1 = np.exp(-r * t) * x_0  # energy decay during the period
        x_1 = x_1 + 1  # stimulation at the arrival time
        tau = tau + t
        n_tau = n_tau + 1
        x_0 = x_1
    return tau, n_tau


# ----------------   main program, set up, make paths, analyse Monte Carlo data
seed = 12344  # change this to get different answers
bg = PCG64(seed)  # instantiate a bit generator
rg = Generator(bg)  # instantiate a random number generator with this bit generator

F_set = [0.5, 3]  # parameter sets
Lambda_set = [0.5, 5]
r_set = [0.5, 5]
N = 10000  # number of samples per MC estimate

print(
    "parameters set          average_tau   error_bar_tau   MonteCarlo_estimate_tau      average_n_tau   error_bar_n_tau    MonteCarlo_estimate_n_tau     computer_time\n")
for F in F_set:
    for Lambda in Lambda_set:
        for r in r_set:
            # for possible combinations of parameters
            sum_tau = 0  # sum of all tau
            sum_n_tau = 0  # sum of all n_tau
            sum_tau_sq = 0  # sum of tau square
            sum_n_tau_sq = 0  # sum of n_tau square
            start_time = time.process_time()  # record the cpu time
            for p in range(N):
                tau, n_tau = mc_cellfire(F, Lambda, r, rg)  # simulate a path
                # compute the sum of interested items
                sum_tau += tau
                sum_n_tau += n_tau
                sum_tau_sq += tau ** 2
                sum_n_tau_sq += n_tau ** 2
            # estimation for tau
            time_eclipse = time.process_time() - start_time  # return cpu time
            tau_hat = sum_tau / N  # sample mean for tau
            stau_hat = np.sqrt(sum_tau_sq / N - tau_hat ** 2)  # estimate the std for tau
            eb_tau = stau_hat / np.sqrt(N)  # error bar
            # estimation for n_tau, the following procedure is the same as the above
            n_tau_hat = sum_n_tau / N
            sn_tau_hat = np.sqrt(sum_n_tau_sq / N - n_tau_hat ** 2)
            eb_n_tau = sn_tau_hat / np.sqrt(N)
            out = "({F:3.1f}, {Lambda:3.1f}, {r:3.1f})        {tau:10.3e}    {eb_tau:10.3e}  {tau:14.3e}(\u2213{eb_tau:7.2e})    {ntau:15.3e}    {eb_ntau:12.3e}  {ntau:17.3e}(\u2213{eb_ntau:7.2e})    {time: 15.3e}"
            out = out.format(F=F, Lambda=Lambda, r=r, tau=tau_hat, eb_tau=eb_tau, ntau=n_tau_hat, eb_ntau=eb_n_tau,
                             time=time_eclipse)
            print(out)
