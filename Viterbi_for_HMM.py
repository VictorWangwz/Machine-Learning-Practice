__author__ = ' Zhen Wang'

import numpy as np


class ViterbiDecoding:
    def __init__(self, pt, p0, pe, o):
        self.pt = pt
        self.p0 = p0
        self.pe = pe
        self.o = o
        self.path = []

    def decode(self):
        n_o = self.o.shape[0]
        p = self.p0 * self.pe[:,self.o[0]]
        index = np.argmax(p)
        p_new = np.max(p)
        self.path.append(index)

        for i in range(1, n_o):
            p = p_new * self.pt[index, :] * self.pe[:, self.o[i]]
            print(p)
            index = np.argmax(p)
            p_new = np.max(p)
            self.path.append(index)
        print(self.path)




if __name__ == "__main__":
    pt = np.asarray(
            [
                [0.5, 0.2, 0.3],
                [0.3, 0.5, 0.2],
                [0.2, 0.3, 0.5]
            ]
        )
    p0 = np.asarray([0.2, 0.4, 0.4]).transpose()
    pe = np.asarray([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])
    o = np.asarray([0,1,0])
    decoder = ViterbiDecoding(pt, p0, pe, o)
    decoder.decode()
