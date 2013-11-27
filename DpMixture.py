
from distributions import *
from GaussCompnt import *

class DpMixture:
    def __init__(self, compnt_model, hyparams, alpha):
        # Other than Gaussian, component model can be any other models
        # with corresponding hyperparameters
        self.compnt_model = compnt_model
        self.hyparams     = hyparams
        self.data         = []
        self.assigns      = {}
        self.comps        = []
        self.alpha        = alpha

    def add_point(self, x):
        self.data.append(x)

    def __sample_z(self, i):
        x = self.data[i]
        if i in self.assigns:
            k = self.assigns[i]
            # Delete empty component
            if self.comps[k].del_point(x)==0:
                del self.comps[k]
                for j,v in self.assigns.items():
                    self.assigns[j] -= int(v>k)

        # Compute parameters for discrete distribution 
        nc = len(self.comps)
        pp = [self.comps[k].n * self.comps[k].posterior_pdf(x) for k in range(nc)]
        
        new_comp = self.compnt_model(self.hyparams) 
        pp.append(self.alpha * new_comp.posterior_pdf(x)) 
        
        j = Discrete(pp)
        self.assigns[i] = j

        if j>=nc:
            self.comps.append(new_comp)

        self.comps[j].add_point(x)

    def gibbs(self, it=1):
        for step in range(it):
            for i in range(len(self.data)-1, -1, -1):
                self.__sample_z(i)

    def pdf(self, X):
        # Draw mixing proportions over active and inactive components
        aa = [c.n for c in self.comps]
        aa.append(self.alpha)
        pp = list(Dirichlet(aa))

        # Further draw mixing proportions over inactive components via Stick-breaking
        tolerance = 0.01
        while pp[-1] > tolerance:
            b = Beta(1, self.alpha)
            k = len(pp)
            pp[k-1:k+1] = [pp[-1]*b, pp[-1]*(1-b)] 
        
        new_comp = self.compnt_model(self.hyparams)
        Y = [0]*len(X)
        for k in range(len(pp)):
            if k < len(self.comps):
                Y += pp[k] * self.comps[k].posterior_pdf(X)
            else:
                Y += pp[k] * new_comp.posterior_pdf(X)
        return Y

