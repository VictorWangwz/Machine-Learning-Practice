__author__ = ' Zhen Wang'
import numpy as np
import matplotlib.pyplot as plt

# When the prob p(x) is too complex that it is hard to directly sample with this prob
# Try rejection sample


def sample_prob():
    return np.random.rand() * 6


def get_target_prob(x):
    return 1/4 * np.exp(-(x-1) ** 2) + 3/4 * np.exp(-(x - 3) ** 2 )


def rejection_sample(iter=100000):
    samples = np.zeros(iter)
    k = 1
    for i in range(iter):
        accept = False
        # If the random sample is under the line of the target
        # accept it, otherwiser reject it.
        while not accept:
            x = sample_prob()
            # evenly distribution above the target one
            u = np.random.rand() * k * 6
            if u < get_target_prob(x):
                samples[i] = x
                accept = True
    return samples


if __name__ ==  "__main__":
    x = np.arange(0, 6, 0.1)
    samples = rejection_sample()
    plt.hist(samples, 15, normed=1, fc='c')
    plt.plot(x, 1/4 * np.exp(-(x-1) ** 2) + 3/4 * np.exp(-(x - 3) ** 2 ), 'g', lw=3)
    plt.axis([-0.5, 6, 0, 1])
    plt.show()