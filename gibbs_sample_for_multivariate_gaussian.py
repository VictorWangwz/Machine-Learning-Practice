__author__ = ' Zhen Wang'

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


# user p(x,y) to compute p(x|y) given random y
def p_x_cond_y(
        y,
        mu_x, mu_y,
        sigma_xx, sigma_xy, sigma_yx, sigma_yy
):
    mu_x_cond_y = mu_x + sigma_xy / sigma_yy *( y - mu_y )
    sigma_x_cond_y = sigma_xx - sigma_xy / sigma_yy *sigma_yx
    return np.random.normal(mu_x_cond_y, sigma_x_cond_y)


# sampling for given iter; in each iter in turn sample each dim (x, y)
def gibbs_sampling_for_m_gaussian(mu, sigma, iter=100000):
    samples = np.zeros((2, iter))
    x = 10 * np.random.rand()

    for i in range(iter):
        y = p_x_cond_y(
            x,
            mu[1], mu[0],
            sigma[0,0], sigma[0, 1], sigma[1, 0], sigma[1, 1]
        )
        x = p_x_cond_y(
            y,
            mu[0], mu[1],
            sigma[1,1], sigma[1, 0], sigma[0, 1], sigma[0, 0]
        )
        samples[:, i] = [x, y]

    return samples


if __name__ == "__main__":
    mu = np.array([1, 1])
    sigma = np.array([[1, 0], [0, 1]])
    samples = gibbs_sampling_for_m_gaussian(mu, sigma)
    sns.jointplot(samples[0, :], samples[1,:])
    plt.show()