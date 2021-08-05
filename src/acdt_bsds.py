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
    # BSDS
    n = 5000
    k = 30
    l = 5
    d = 64
    X = load_bsds('../data/BSDS300/images/train/', n)
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    total = time.time()
    acdt = ACDT(k, l, d, X, minimum_ckpt=300, store_every=5, visualize=False)
    acdt.fit()
    print('Took: %ss' % (time.time() - total))

    PATH = './saved/'
    os.makedirs(PATH, exist_ok=True)
    with open(os.path.join(PATH, 'ckpt_bsds_%s.pickle' % str(d)), 'wb') as f:
        pickle.dump(acdt.checkpoints, f, protocol=pickle.HIGHEST_PROTOCOL)
