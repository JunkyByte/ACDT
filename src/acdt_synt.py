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
    datasets_3d = {'swiss': datasets.make_swiss_roll, 'scurve': datasets.make_s_curve}
    for name, dataset in datasets_3d.items():
        print('Dataset', name)
        n = 5000
        k = 15
        l = 2
        d = 2
        X, _ = dataset(n)
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

        total = time.time()
        acdt = ACDT(k, l, d, X, minimum_ckpt=100, store_every=0, visualize=False)
        acdt.fit()
        print('Took: %ss' % (time.time() - total))
        acdt.pool.close()

        PATH = './saved/'
        os.makedirs(PATH, exist_ok=True)
        file_name = 'ckpt_' + name + '.pickle'
        with open(os.path.join(PATH, file_name), 'wb') as f:
            pickle.dump(acdt.checkpoints, f, protocol=pickle.HIGHEST_PROTOCOL)
        # draw_3d_clusters(acdt.C, X)
