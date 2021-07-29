import os
import pickle
import time
import numpy as np
import sklearn.datasets as datasets
from datasets_util import make_spiral, make_2_spiral, load_bsds
from draw_utils import draw_3d_clusters, draw_spiral_clusters
from acdt import ACDT
np.random.seed(42)


if __name__ == '__main__':
    # BSDS
    # n = 10000
    # k = 100
    # l = 15
    # d = 16
    # X = load_bsds('../data/BSDS300/images/train/', n)
    # X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # S curve and Swiss roll
    n = 5000
    k = 15
    l = 2
    d = 2
    X, _ = datasets.make_s_curve(n)
    # X, _ = datasets.make_swiss_roll(n)
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # 2d datasets
    # n = 500
    # k = 4
    # l = 2
    # d = 1
    # X = make_spiral(n=n)
    # X = make_2_spiral(n=n)
    # X, _ = datasets.make_circles(n)

    total = time.time()
    acdt = ACDT(k, l, d, X, minimum_ckpt=300, store_every=5, visualize=False)
    acdt.fit()
    print('Took: %ss' % (time.time() - total))

    PATH = './saved/'
    os.makedirs(PATH, exist_ok=True)
    with open(os.path.join(PATH, 'ckpt.pickle'), 'wb') as f:
        pickle.dump(acdt.checkpoints, f, protocol=pickle.HIGHEST_PROTOCOL)

    # if X.shape[1] == 2:
    #     draw_spiral_clusters(acdt.C, k)
    # if X.shape[1] == 3:
    #     draw_3d_clusters(acdt.C)
