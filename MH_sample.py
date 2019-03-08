__author__ = ' Zhen Wang'
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm

class MCMC:
    def __init__(self):
        self.p0 = np.array([[0.1, 0.6, 0.3, 0.0, 0.0, 0.0, 0.0]])
        self.pt = np.array(
            [
                [0.08, 0.9, 0.01, 0.0, 0.0, 0.0, 0.01],
                [0.03, 0.95, 0.01, 0.0, 0.0, 0.0, 0.01],
                [0.06, 0.06, 0.75, 0.05, 0.05, 0.02, 0.01],
                [0.0, 0.0, 0.0, 0.3, 0.6, 0.09, 0.01],
                [0.0, 0.0, 0.0, 0.02, 0.95, 0.02, 0.01],
                [0.0, 0.0, 0.0, 0.01, 0.01, 0.97, 0.01],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            ]
        )
        self.n = self.p0.shape[1]

    # Converge to stationary distribution with init p0 and pt
    def get_pi_x(self, iter=1000):
        p0 = self.p0
        pt = self.pt
        records = [[] for i in range(self.n)]

        for i in range(iter):
            p0 = np.dot(p0, pt)
            for j in range(self.n):
                records[j].append(p0[0][j])
        x = np.arange(iter)
        for i in range(self.n):
            plt.plot(x,records[i],label=i)
        plt.legend()
        plt.show()

    # pt Coverge with iteration
    def get_pt_n(self, iter=10):
        pt = self.pt
        records = [[] for i in range(self.n*self.n)]

        for i in range(iter):
            pt = np.dot(pt, pt)
            for j in range(self.n):
                for k in range(self.n):
                    records[j*self.n+k].append(pt[j][k])
        x = np.arange(iter)
        for i in range(self.n*self.n):
            plt.plot(x, records[i], label='value1')
        plt.show()

    def pi_q(self, theta):
        return norm.pdf(theta, loc=3, scale=2)

    # M H sampling
    def M_H_sample(self):
        num = 5000
        pi = [0 for i in range(num)]
        for t in range(1, num):
            x_star = norm.rvs(loc=pi[t-1], scale=1, size=1, random_state=None)
            u = np.random.rand()
            alpha = min(1, self.pi_q(x_star[0])/self.pi_q(pi[t-1]))
            if u < alpha:
                pi[t] = x_star[0]
            else:
                pi[t] = pi[t-1]
        plt.scatter(pi, norm.pdf(pi, loc=3, scale=2))
        num_bins = 50
        plt.hist(pi, num_bins, normed=1, alpha=0.7)
        plt.show()



if __name__  == "__main__":
    sampler = MCMC()
    sampler.M_H_sample()