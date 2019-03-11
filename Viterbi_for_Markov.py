__author__ = ' Zhen Wang'

import numpy as np
from matplotlib import pyplot as plt


class ViterbiForMarkov:
    def __init__(self, p0, pt):
        self.p0 = p0
        self.pt = pt

    def decode(self, d=50):
        p0 = self.p0
        t = self.p0.shape[0]
        p = np.zeros((t, d))
        index = np.zeros((t, d))
        for i in range(d):
            new_p = (p0* self.pt.transpose()).transpose()
            p0 = np.amax(new_p, axis=0)
            index[:, i] = np.argmax(new_p, axis=0)
            p[:, i]  = p0
        print(index)
        x = np.arange(d)
        for i in range(t):
            plt.plot(x, p[i], label='value1')
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
    decoder = ViterbiForMarkov(p0, pt)
    decoder.decode(d=100)