__author__ = ' Zhen Wang'
import numpy as np
import marginalCK
from matplotlib import pyplot as plt


class BackwardsSample:
    def __init__(self, p0, pt):
        self.p0 = p0
        self.pt = pt

    def cdf(self, p):
        cdf = np.copy(p)
        for i in range(1, cdf.shape[0]):
            cdf[i] = cdf[i] + cdf[i-1]
        return cdf

    def sample(self, xd, t=1000, d=50):
        ck = marginalCK.MarginalCK(self.p0, self.pt)
        p = ck.ck(d)
        mcs = np.zeros((t, d))
        for i in range(t):
            p0 = np.copy(self.pt[:, xd] * p[:, d-1] / sum(self.pt[:, xd] * p[:,d-1]))
            mcs[i, d-1] = xd
            for j in range(d-2, -1, -1):
                cdf = self.cdf(p0)
                r = np.random.rand()
                index = np.argwhere(cdf >= r)[0][0]
                mcs[i, j] = index
                p0 = np.copy(self.pt[:, index]*p0/sum(self.pt[:, index]*p0))
        return mcs

    def monte_carlo(self, samples):
        num_sample = samples.shape[0]
        n = self.p0.shape[0]
        d = samples.shape[1]
        marginal_prob = np.zeros([n, d])
        for i in range(d):
            for j in range(n):
                marginal_prob[j,i] = np.argwhere(samples[:, i]==j).shape[0]/num_sample
        print(marginal_prob)
        x = np.arange(d)
        for i in range(n):
            plt.plot(x, marginal_prob[i,:], label='value1')
        plt.show()



if __name__ == "__main__":
    p0 = np.asarray([0.1, 0.6, 0.3, 0, 0, 0, 0])
    pt = np.asarray(
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
    backwards_sample = BackwardsSample(p0, pt)
    mcs = backwards_sample.sample(xd=1)
    backwards_sample.monte_carlo(mcs)