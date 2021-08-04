import os
import pickle
import time
import numpy as np
import sklearn.datasets as datasets
from datasets_util import make_spiral, make_2_spiral, load_bsds, load_vidtimit, load_mnist
from draw_utils import draw_3d_clusters, draw_spiral_clusters
from acdt import ACDT
np.random.seed(42)


if __name__ == '__main__':
    n = 2000
    k = 15
    l = 5
    d = 100

    digit = 8
    print('PROCESSING DIGIT %s' % digit)
    X = load_mnist('../data/MNIST/', digit=digit, n=n)
    X = X / 255

    total = time.time()
    acdt = ACDT(k, l, d, X, minimum_ckpt=50, store_every=1, visualize=False)
    acdt.fit()
    print('Took: %ss' % (time.time() - total))

    PATH = './saved/'
    os.makedirs(PATH, exist_ok=True)
    file_name = 'ckpt_mnist_%s.pickle' % digit
    with open(os.path.join(PATH, file_name), 'wb') as f:
        pickle.dump(acdt.checkpoints, f, protocol=pickle.HIGHEST_PROTOCOL)
