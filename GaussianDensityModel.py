__author__ = ' Zhen Wang'
from math import exp, log, pi

import h5py
import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as LA
from numpy.linalg import inv
from numpy.linalg import multi_dot


class GDM:
    def __init__(self, x):
        self.initiate(x)

    def initiate(self, x):
        (d, n) = x.shape
        self.mu = np.zeros((2, 1))
        self.mu = np.sum(X, axis=1)/n
        subtaction = np.subtract(np.transpose(X), np.transpose(self.mu))
        self.Sigma = 1/n * np.dot(np.transpose(subtaction), subtaction)
        self.SigmaInv = inv(self.Sigma)

    def pdf(self, X):
        (d, n) = X.shape
        pdf = np.zeros(n)
        for i in range(n):
            pdf[i] = exp(
                -0.5*multi_dot([
                    np.transpose(X[:, i] - self.mu),
                    self.SigmaInv,
                    X[:, i] - self.mu
                ])
                - d/2 * log(2*pi, 10)
                -1/2 *log(LA.norm(self.Sigma), 10)
            )
        return pdf



if __name__ == "__main__":
    f = h5py.File("mixtureData.jld", "r");
    X = f["X"].value[1,:].reshape((1, 250))
    Xhat = np.asarray(range(-125, 125)).reshape((1, 250))
    gdm = GDM(X)
    n = X.shape[1]
    x = np.arange(250)
    plt.plot(x, gdm.pdf(Xhat), label='value1')
    plt.show()
