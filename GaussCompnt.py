
from distributions import *

class GaussCompnt:
    def __init__(self, (mu0, beta0, a0, b0)):
        self.mu0   = mu0
        self.beta0 = beta0
        self.a0    = a0
        self.b0    = b0
        self.n     = .0
        self.sumx  = .0
        self.sumx2 = .0

    def add_point(self, x):
        self.n     += 1
        self.sumx  += x
        self.sumx2 += x*x
        return self.n
    
    def del_point(self, x):
        assert self.n > 0
        self.n     -= 1
        self.sumx  -= x
        self.sumx2 -= x*x
        return self.n
  
    def posterior(self):
        assert self.n >= 0
        
        if self.n == 0:
            return Gaussian_Gamma(self.mu0, self.beta0, self.a0, self.b0)

        mu   = (self.beta0 * self.mu0 + self.sumx) / (self.beta0 + self.n)
        beta = self.beta0 + self.n

        a    = self.a0 + self.n/2.0
        b    = self.b0 + (self.sumx2 - self.sumx ** 2 / self.n + self.beta0 * (self.sumx - self.n * self.mu0) ** 2 / (self.beta0 + self.n))/2.0

        return Gaussian_Gamma(mu, beta, a, b)
	
    def posterior_pdf(self, X):
        m, l = self.posterior()
        return Gaussian_pdf(X, m, l)

