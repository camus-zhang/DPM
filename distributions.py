import numpy.random as random
import scipy.stats as ss

def Dirichlet(a):
    return random.dirichlet(a)

def Beta(a, b):
    return random.beta(a, b)

def Gaussian_Gamma(mu0, beta0, a0, b0):
    l = random.gamma(a0, b0)
    m = random.normal(mu0, 1.0/beta0/l)
    return m,l

def Discrete(a):
    r = random.random()*sum(a)
    cum_sum = [sum(a[:i+1]) for i in range(len(a))]
    return sum([r>e for e in cum_sum])

def Gaussian_pdf(X, m, l):    
	return ss.norm.pdf(X,m,1.0/l)
