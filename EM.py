__author__ = ' Zhen Wang'
import numpy as np
import h5py
from numpy import linalg as LA
from numpy.linalg import inv
from numpy.linalg import multi_dot
import numpy as np
from matplotlib import pyplot as plt
from math import exp, log, pi


class EM:
    def __init__(self, X, k=2):
        self.initiate(X, k)

    def initiate(self, x, k):
        self.k = k
        (d, n) = x.shape
        self.mu = np.random.rand(k, d)
        self.Sigma = np.random.rand(k, d, d)
        self.SigmaInv = np.random.rand(k, d, d)
        self.pc = 1/k * np.ones(k)

    def pdf(self, X, c):
        (d, n) = X.shape
        pdf = np.zeros(n)
        for i in range(n):
            pdf[i] = exp(
                -0.5*multi_dot([
                    np.transpose(X[:, i] - self.mu[c, :]),
                    self.SigmaInv[c, :, :],
                    X[:, i] - self.mu[c, :]
                ])
                - d/2 * log(2*pi, 10)
                -1/2 *log(LA.norm(self.Sigma[c, :, :]), 10)
            )
        return pdf

    def pdfs(self, X):
        (d, n) = X.shape
        pdf = np.zeros(n)
        pdfs = np.zeros((self.k, n))
        for c in range(self.k):
            pdfs[c] = pdf(X, c)

        for i in range(n):
            for c in range(self.k):
                pdf[i] += self.pc[c]*pdfs[c]
        return pdf

    def em(self, X, iter=100):
        (d, n) = X.shape
        for i in range(iter):
            rc = np.zeros((n, self.k))
            for c in range(self.k):
                rc[:,c] = self.pdf(X,c) * self.pc[c] / self.pdfs(X)
                self.pc = 1/n * np.sum(self.pc, axis=1)
            for c in range(self.k):
                self.mu[c, :] = np.dot(rc[:, c], X) / np.sum(rc[:, c])


if __name__ == "__main__":
    f = h5py.File("mixtureData.jld", "r");
    X = f["X"].value[1, :].reshape((1, 250))
    em = EM(X)