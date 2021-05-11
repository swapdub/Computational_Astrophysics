# Midterm Astro 410
# Swapnil Dubey
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *

# Import data as Pandas DataFrame
df = pd.read_csv("midterm.dat", sep="\s+", header=None)

# Setting random seed value for consistent results over multiple runs
seed(4)

#Def Function for our Gaussian 
def G(x, mu, alpha_D, A):
    return ((A / alpha_D) * np.sqrt(np.log(2) / np.pi) * np.exp(-1 * np.log(2) * (x-mu)**2 / alpha_D**2))

def P(y_i, m_i, sigma_i):
    return (q/sigma_i * np.sqrt(2 * np.pi)) * np.exp(-1 * (y_i-m_i)/ sigma_i * np.sqrt(2))

# Use Guess values to plot and manually choose good starting guess for fitting curve 
# Guess as dict
guess = {'x':43, 'mu':44.954, 'alpha_D':15.027, 'A':1}
# Guess as array; v_0: alpha_L
guess1 = [43, 8]


# Calculating output with our guessed inputs so we can build plot
n = len(df[1])
y = np.empty(n)

for i in range(n):
    y[i] = G(df[0][i],guess['mu'], guess['alpha_D'], guess['A'])

plt.errorbar(df[0], df[1], fmt='.')
plt.title("Raw Data Plot")
plt.errorbar(df[0], y, fmt='--')
plt.ylabel("Line Strength \n \u03C6(\u03BD)")
plt.xlabel("Frequency \n \u03BD")
plt.legend(['Original Data', 'Gausian Fit'])
plt.show()


def gaussian(x_i, p):
    a0 = 1.177410225 #sqrt(2ln(2))
    a1 = 0.3989423 #1/sqrt(2*pi)
    s = p[1]/a0
    numer = (x_i-p[0])/s
    m = a1*p[2]/s * np.exp(-0.5*numer**2)
    return m

#this computes the -ln(likelyhood)
def mlnlikely(p, d):
    xd = d[0]
    yd = d[1]
    sd = d[2]
    m = gaussian(xd, p)
    r = 0.5*((yd-m)/sd)**2 # ignore the factor 1/sqrt(2*pi)/sd, because it does not depend on the model parameters.
    return sum(r)

def mcmc(d, p0, s0, nm):
    np = len(p0)
    nq = np+2
    #first dim of p is parameter index, 2nd dim is the chain iteration index
    #p[0:np,i] store the np parameters of i-th iteration
    #p[np,i] store the -ln(likelyhood) of i-th iteration
    #p[np+1,i] store the rejection probability i-th iteration
    p = zeros((nq,nm))
    #copy the inital parameter into p[:np,0]
    for ip in range(np):
        p[ip,0] = p0[ip]
    #compute the -ln(likelyhood) for the initial parameters
    p[np,0] = mlnlikely(p0, d)
    #iterate along the chain
    for i in range(1,nm):
        #random jump from i-1 iteration to new parameters pnew
        #x is the np random variables uniformly distributed from -1 to 1
        x = 2*random(np)-1.0
        pnew = zeros(np)
        for ip in range(np):
            pnew[ip] = p[ip,i-1] + x[ip]*s0[ip]
        #compute the -ln(likelyhood) for the new parameters, store in p[np,i]
        p[np,i] = mlnlikely(pnew, d)
        if p[np,i] <= p[np,i-1]:
            #if p[np,i] <= p[np,i-1], the new parameters are accepted
            for ip in range(np):
                p[ip,i] = pnew[ip]
        else:
            #if p[np,i] > p[np,i-1], the new parameters are accepted with probability of r = exp(-(p[np,i]-p[np,i-1]))
            r = exp(-(p[np,i]-p[np,i-1]))
            #1-r is the rejection probability, store in p[np+1,i]
            p[np+1,i] = 1-r
            y = random()
            if (y > r):
                #reject pnew, copy p[:,i-1] into p[:,i]
                for ip in range(np+1):
                    p[ip,i] = p[ip,i-1]
            else:
                #accept pnew, copy pnew into p[:,i]
                for ip in range(np):
                    p[ip,i] = pnew[ip]
                
        if (i+1)%100 == 0:
            print('%6d %10.3E %10.3E %10.3E %10.3E %10.3E %10.3E'%(i,p[np+1,i],p[np,i],p[0,i],p[1,i],p[2,i],mean(p[np+1,:i+1])))
    return p


# Initial guess for Parameters
#     mu,  alpha_D, A
guess = [40, 15, 2]

# Optimal MCMC widths
# s0 = [0.06, 0.04, 0.0025]
s0 = [0.06, 0.04, 0.0025]

# Number of Iterations
nm = 10_000

# Gives us probability for each Parameter
p = mcmc(df, guess, s0, nm)


# Histogram showing the frequency of occurance for a given value range in the predicted markov chain
plt.hist(p[0], bins=20, density=True, stacked=True)
plt.title("Probability Distribution for \u03BC")
plt.ylabel("Number of Occurences")
plt.xlabel("Predicted Values of \u03BC")
plt.show()

plt.hist(p[1], bins=20, density=True, stacked=True)
plt.title("Probability Distribution for \u03B1_D")
plt.ylabel("Number of Occurences")
plt.xlabel("Predicted Values of \u03B1_D")
plt.show()

plt.hist(p[2], bins=20, density=True, stacked=True)
plt.title("Probability Distribution for 'A'")
plt.ylabel("Number of Occurences")
plt.xlabel("Predicted Values of 'A'")
plt.show()


# Graph Estimating parameter mu
plt.errorbar(range(0,10000), p[0], fmt='')
plt.title("MCMC \u03BC History")
plt.ylabel("Predicted value of \u03BC")
plt.xlabel("Number of iterations")
plt.show()

# Graph Estimating parameter Alpha_D
plt.errorbar(range(0,10000), p[1], fmt='')
plt.title("MCMC \u03B1_D History")
plt.ylabel("Predicted value of \u03B1_D")
plt.xlabel("Number of iterations")
plt.show()

# Graph Estimating parameter A
plt.errorbar(range(0,10000), p[2], fmt='')
plt.title("MCMC 'A' History")
plt.ylabel("Predicted value of 'A'")
plt.xlabel("Number of iterations")
plt.show()