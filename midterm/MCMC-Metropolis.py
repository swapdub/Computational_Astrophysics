from pylab import *

"""
usage: python mcmc.py
"""

#this computes the Gaussian fit function
def gaussian(xd, p):
    a0 = 1.177410225 #sqrt(2ln(2))
    a1 = 0.3989423 #1/sqrt(2*pi)
    s = p[1]/a0
    dx = (xd-p[0])/s
    m = a1*p[2]/s * np.exp(-0.5*dx*dx)
    return m

#this computes the -ln(likelyhood)
def mlnlikely(p, d):
    xd = d['xd']
    yd = d['yd']
    sd = d['sd']
    m = gaussian(xd, p)
    r = 0.5*((yd-m)/sd)**2 # ignore the factor 1/sqrt(2*pi)/sd, because it does not depend on the model parameters.
    return sum(r)
"""
d contains the data
p0 is the intial parameters
s0 is the width of the proposal distribution
nm is the number of iterations in the Markov Chain
"""
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

