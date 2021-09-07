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
    # VidTIMITT
    subject = 2
    k = 15
    l = 2
    d = 10
    X = load_vidtimit('../data/vidtimit/', subject=subject)
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    n = len(X)
    print('Loaded vidtimit subject %s with size %s' % (subject, n))

    total = time.time()
    acdt = ACDT(k, l, d, X, minimum_ckpt=100, store_every=2, visualize=False)
    acdt.fit()
    print('Took: %ss' % (time.time() - total))

    PATH = './saved/'
    os.makedirs(PATH, exist_ok=True)
    with open(os.path.join(PATH, 'ckpt_vidtimit_%s.pickle' % subject), 'wb') as f:
        pickle.dump(acdt.checkpoints, f, protocol=pickle.HIGHEST_PROTOCOL)

    # if X.shape[1] == 2:
    #     draw_spiral_clusters(acdt.C, k)
    # if X.shape[1] == 3:
    #     draw_3d_clusters(acdt.C)
