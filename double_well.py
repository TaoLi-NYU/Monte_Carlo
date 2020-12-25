"""
Code for Assignment 6, MonteCarlo methods, Nov 10, 2020
by Tao Li, taoli@nyu.edu

"""
import numpy as np
from numpy.random import Generator, PCG64  # numpy random number generator
from scipy.stats import norm
import matplotlib.pyplot as plt


def rho(x, alpha, beta):
    """
    The pdf of target distribution
    input: x: sample point, alpha: parameter in the pdf, beta: inverse temperature
    return: the prob of x
    """
    phi = np.square(x * x - 1) + alpha * x
    prob = np.exp(-beta * phi)
    return prob


def prop(y, r, rg):
    """
        Proposal
        input: y: current sample, r: proposal size std dev
        return: another sample from N(y, r^2 I)
    """
    x = rg.normal(y, r)
    return x


def rsample(y, r, alpha, beta, rg):
    """
    resampling
    input: y: current sample, r: proposal size, alpha,beta as defined in rho
    return: new sample and 0: if y is retained, 1: if a new x is generated
    """
    # have a proposal
    x = prop(y, r, rg)
    # evaluate the pdf of target
    rhoX = rho(x, alpha, beta)
    rhoY = rho(y, alpha, beta)
    # evalute the pdf of the proposal
    PYX = norm(x, r).pdf(y)
    PXY = norm(y, r).pdf(x)
    # metropolis
    MH = (rhoX * PYX) / (rhoY * PXY)
    if MH > 1:
        return x, 1
    U = rg.random()
    if U < MH:
        return x, 1
    return y, 0


def MHMC(x0, r, alpha, beta, N, rg):
    """
    metropolis
    input: x0: initial sample, r, alpha, beta: as defined above N: length
    return: sample path
    """
    # allocation
    path = np.empty(N)
    accept_num = 0
    # total: the number of all samples generated
    # counter: the number of valid samplers
    for i in range(N):
        path[i] = x0
        # MH sampler for one move
        x1, accept = rsample(x0, r, alpha, beta, rg)
        x0 = x1
        accept_num = accept_num + accept

    accept_rate = accept_num / N
    return path, accept_rate


def autocov_est(v_arr, t):
    """
    Estimate the auto-covariance
    input: v_arr: sample path, t: time variable
    """
    length = len(v_arr)
    if t > length - 1:
        print('invalid t')
        return
    v_bar = v_arr.mean()
    v_diff0 = v_arr[0:length - 1 - t] - v_bar
    v_difft = v_arr[t:length - 1] - v_bar
    est = np.multiply(v_diff0, v_difft).mean()
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


def para_experiment(alpha, beta, r, rg, T=800, N=15000, plot=True, x0=0.1):
    """
    numerical experiment for different combinations of parameters
    input: alpha, beta, r: parameters defined above
        rg: random generator, T: maximum time , N: length of sample path,
        plot: if true, output the plots, x0: the initial point
    output: path: sample path, rate: acceptance rate  tau: auto-correlation time for X,
        tau_sq: auto-correlation time for X^2
    """
    # get the sample path
    path, rate = MHMC(x0, r, alpha, beta, N, rg)
    path_sq = np.power(path, 2)
    # estimate the auto-correlation function
    autocorr = auto_corr(path, T)
    autocorr_sq = auto_corr(path_sq, T)
    # estimate the auto-correlation time
    tau = auto_time(autocorr, 5)
    tau_sq = auto_time(autocorr_sq, 5)
    # get plots
    if plot == True:
        plt.plot(np.linspace(0, N, N), path)
        plt.xlabel("time k")
        plt.ylabel("X")
        title_text = "Sample path under alpha={alpha:.2f} , beta={beta:.2f} , r={r:.2f} "
        plt.title(title_text.format(alpha=alpha, beta=beta, r=r))
        plt.show()
        #
        plt.plot(np.linspace(0, N, N), path_sq)
        plt.xlabel("time k")
        plt.ylabel("X^2")
        title_text = "Sample squared path under alpha={alpha:.2f} , beta={beta:.2f} , r={r:.2f} "
        plt.title(title_text.format(alpha=alpha, beta=beta, r=r))
        plt.show()
        #
        plt.plot(np.linspace(0, T, T), autocorr, label='X')
        plt.plot(np.linspace(0, T, T), autocorr_sq, alpha=0.5, label='X^2')
        plt.xlabel("time t")
        plt.ylabel("auto correlation")
        plt.legend(loc='upper right')
        title_text = "auto correlation function under alpha={alpha:.2f} , beta={beta:.2f} , r={r:.2f} "
        plt.title(title_text.format(alpha=alpha, beta=beta, r=r))
        plt.show()
        #
        plt.hist(path, bins='auto', alpha=1, label='global')
        title_text = "Histogram under alpha={alpha:.2f} , beta={beta:.2f} , r={r:.2f} "
        plt.title(title_text.format(alpha=alpha, beta=beta, r=r))
        plt.show()
    return path, rate, tau, tau_sq


# main program
seed = 12344  # change this to get different answers
bg = PCG64(seed)  # instantiate a bit generator
rg = Generator(bg)  # instantiate a random number generator with this bit generator

# experiment with various parameters

# plot for fig 1, uncomment to recover the figure
para_experiment(alpha=0, beta=300, r=1, rg=rg, T=800)
# plot for fig 2
# para_experiment(alpha=0, beta=300, r=1, rg=rg, T=2000)
# plot for fig 3
# para_experiment(alpha=0.5, beta=300, r=1, rg=rg, T=2000)
# plot for fig 4
# para_experiment(alpha=0, beta=300, r=1, rg=rg, T=2000, N=15000)
# para_experiment(alpha=0.5, beta=300, r=1, rg=rg, T=2000, N=15000)
# para_experiment(alpha=1, beta=300, r=1, rg=rg, T=2000, N=15000)
# para_experiment(alpha=2, beta=300, r=1, rg=rg, T=2000, N=15000)
# plot for fig 5
# para_experiment(alpha=0, beta=100, r=0.2, rg=rg, T=1500, N=15000)
# para_experiment(alpha=0, beta=100, r=1, rg=rg, T=1500, N=15000)
# para_experiment(alpha=0.5, beta=200, r=0.15, rg=rg, T=1500, N=15000)
# para_experiment(alpha=0.5, beta=200, r=1, rg=rg, T=1500, N=15000)

# table 1
# para_experiment(alpha=0.5, beta=50, r=0.3, rg=rg, T=1500, N=15000)
# para_experiment(alpha=0.5, beta=100, r=0.2, rg=rg, T=1500, N=15000)
# para_experiment(alpha=0.5, beta=150, r=0.16, rg=rg, T=1500, N=15000)
# para_experiment(alpha=0.5, beta=200, r=0.15, rg=rg, T=1500, N=15000)
