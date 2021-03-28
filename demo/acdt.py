import numpy as np

class ACDT:
    def __init__(self, k, l, d):
        """
        k: number of neighbors during knn.
        l: number of target clusters.
        d: subspace dimensional size
        """
        self.k = k
        self.l = l
        self.d = d

    def fit(self, x):
        pass
