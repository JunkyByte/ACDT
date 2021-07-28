import os
import pickle
import time
import numpy as np
import sklearn.datasets as datasets
from datasets_util import make_spiral, make_2_spiral
from draw_utils import draw_3d_clusters, draw_spiral_clusters
from acdt import ACDT


if __name__ == '__main__':
    # Params
    k = 15
    l = 12
    d = 2

    # dataset points
    n = 5000
    # X = make_spiral(n=n)
    # X = make_2_spiral(n=n)
    # X, _ = datasets.make_circles(n)
    # X, _ = datasets.make_swiss_roll(n)
    X, _ = datasets.make_s_curve(n)

    if X.shape[1] == 3:
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    total = time.time()
    acdt = ACDT(k, l, d, X, minimum_ckpt=100, store_every=1, visualize=False)
    acdt.fit()
    print('Took: %ss' % (time.time() - total))

    PATH = './saved/'
    os.makedirs(PATH, exist_ok=True)
    with open(os.path.join(PATH, 'ckpt.pickle'), 'wb') as f:
        pickle.dump(acdt.checkpoints, f, protocol=pickle.HIGHEST_PROTOCOL)

    if X.shape[1] == 2:
        draw_spiral_clusters(acdt.C, k)
    if X.shape[1] == 3:
        draw_3d_clusters(acdt.C)
