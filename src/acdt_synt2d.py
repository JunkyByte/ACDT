import os
import pickle
import time
import numpy as np
import sklearn.datasets as datasets
from datasets_util import make_spiral, make_2_spiral, load_bsds, load_vidtimit
from draw_utils import draw_3d_clusters, draw_spiral_clusters
from acdt import ACDT
np.random.seed(42)


if __name__ == '__main__':
    # 2d datasets
    n = 500
    k = 4
    l = 2
    d = 1
    X = make_spiral(n=n)
    # X = make_2_spiral(n=n)
    # X, _ = datasets.make_circles(n)

    total = time.time()
    acdt = ACDT(k, l, d, X, minimum_ckpt=100, store_every=1, visualize=False)
    acdt.fit()
    print('Took: %ss' % (time.time() - total))

    PATH = './saved/'
    os.makedirs(PATH, exist_ok=True)
    file_name = 'ckpt_synt2d.pickle'
    with open(os.path.join(PATH, file_name), 'wb') as f:
        pickle.dump(acdt.checkpoints, f, protocol=pickle.HIGHEST_PROTOCOL)
    # draw_spiral_clusters(acdt.C, X, k)
