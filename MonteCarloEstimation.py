#  Code for Monte Carlo class, Jonathan Goodman
#  http://www.math.nyu.edu/faculty/goodman/teaching/MonteCarlo20/
#  MonteCarloEstimation.py
#  The author gives permission for anyone to use this publicly posted 
#  code for any purpose.  The code was written for teaching, not research
#  or commercial use.  It has not been tested thoroughly and probably has
#  serious bugs.  Results may be inaccurate, incorrect, or just wrong.

# Illustrate estimation of a function of a random object (random path)
# by direct simulation and sample averagine with one standard deviation
# error bars.
# See week 1 notes for an explanation in mathematical notation.
# The notation in the code is the same as the notation in the notes.

from numpy.random import Generator, PCG64    #  numpy randon number generator
import numpy as np



#-----------------  simulate (create) a random path X -------------------

def path( X, r, l, rg):
    """generate and return a path with l steps of length r
       X:  A numpy ndarray with shape [2,l+1].  X[:,k] is the 2D location
           of particle k, starting with X[:,0] = (0,0) (the origin).
           Assumed to be allocated already.
           On return, this contains the new path.
           The old path is over-written.
       r:  the ratio of bond length to Debye length
       l:  the number of bonds.  Number of particles is l+1
       rg: a random number generator, already instantiated and seeded."""
       
    X[0,0] = 0.        #   The first particle is at the origin
    X[1,0] = 0.
    
    for k in range(l):
       theta    = 2*np.pi*rg.random()     # random angle, uniform in [0,2*pi]
       dx1      = r*np.cos(theta)         # dx = (dx1,dx2) = the k -> k+1 step
       dx2      = r*np.sin(theta)
       X[0,k+1] = X[0,k] + dx1            # add the step to get the new location
       X[1,k+1] = X[1,k] + dx2            # note: dx1 increments X[0,.]
    return                                # nothing to return, X has changed.
    
    
    
#--------------   function that evaluates V(X) for path X ---------------
    
def Vcalc( X):
    """calculate the Debye interaction energy for the chain
       X:      a random path with l+1 particles
       return: V, the interaction energy"""
       
    [two,lp1] = X.shape           #  determine l from the shape of X
                                  #  the first dimension is 2 = two
                                      
    V = 0.                        #  will be the total interaction potential
    for j in range(l):            #  O(l^2) work -> slow for large l
        for k in range(j+1,l+1):
            R_jk_sq = ( X[0,j] - X[0,k] )**2 + ( X[1,j] - X[1,k] )**2
            R_jk    = np.sqrt( R_jk_sq)
            V       = V + np.exp(-R_jk)    #  Debye potental interaction
    return V
    
    
    
#----------------   main program, set up, make paths, analyse Monte Carlo data
  

seed = 12344             # change this to get different answers
bg   = PCG64(seed)       # instantiate a bit generator
rg   = Generator(bg)     # instantiate a random number generator with ...
                         # ... this bit generator
                          
r = 2.                   # length ratio bond length/Debye length
L = [2, 5, 10, 30, 80]   # lengths of chains
N = 10000                # number of samples per MC estimate

r_string = " r = {r:6.2f}"
r_string = r_string.format(r = r)
print("\n   Direct simulation and sample averaging with " + str(N)
    +     " samples," + r_string + ", and seed "     + str(seed) + "\n")

print(" number of bonds  average V     error bar   Monte Carlo estimate\n")

for l in L:              #  do a calculation for each element of L

    Vsum   = 0.          #  sum of V for all paths with this l
    VsqSum = 0.          #  sum of the squares
    X    = np.ndarray([2,l+1])   #  allocate the path array before making paths
    
    for p in range(N):           #  N samples, accumulate data
    
        path( X, r, l, rg)       #  simulate a new random path
        Vc     = Vcalc(X)        #  evaluate the functional for that path
        Vsum   = Vsum   + Vc     #  compute the sum and ...
        VsqSum = VsqSum + Vc**2  #  ... the sum of the squares
    
    Vhat = Vsum/N                            # sample mean
    sVhat =  np.sqrt( VsqSum/N - Vhat**2)    # estimate stardard deviation of V
    eb    = sVhat / np.sqrt(N)               # one sigma error bar
    out = "    {l:4d}         {Vhat:10.3e}    {eb:10.3e}  {Vhat:10.3e}(\u2213{eb:7.2e})"
    out = out.format(l = l, Vhat = Vhat, eb = eb)
    print(out)
        
    
