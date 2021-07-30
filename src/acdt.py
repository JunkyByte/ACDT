import os
import time
import copy
import numpy as np
import itertools
import scipy.linalg
from numba import njit
from tqdm import tqdm
from karcher_mean import karcher_mean as km
from draw_utils import draw_3d_clusters, draw_spiral_clusters
from sklearn.neighbors import NearestNeighbors
from multiprocessing import Pool


def d_hat(Ci, Cj):
    d_mean = d_geodesic(Ci.M, Cj.M)
    d = (len(Ci) + len(Cj)) * (d_mean ** 2) + 2 * d_mean * (Ci.d + Cj.d)
    return d


def d_geodesic(x, y):
    # Reference: http://dx.doi.org/10.1137%2FS1064827500377332 Page 3
    # This is unstable numerically but it should be fine here
    ua, sa, vha = scipy.linalg.svd(x)
    ub, sb, vhb = scipy.linalg.svd(y)
    ua_smaller = ua[:, :sa.shape[0]]
    ub_smaller = ub[:, :sa.shape[0]]
    QaTQb = np.dot(ua_smaller.T, ub_smaller)
    uQaTQb, sQaTQb, vhQaTQb = scipy.linalg.svd(QaTQb)
    sQaTQb.clip(0, 1, out=sQaTQb)
    thetas = np.arccos(sQaTQb)
    return np.linalg.norm(thetas, ord=2) ** 2


@njit(cache=True)
def argmin_dissimilarity(D, l):
    """
    Finds the argmin on a ndarray D which is supposed to be triu and with -1 for non valid distances
    l can be used to specify the actual size of the valid submatrix (for efficiency)
    """
    n, m = D.shape
    l = m - l
    min_v = np.inf
    min_idx = (-1, -1)
    for i in range(n - l):
        for j in range(i, m - l):
            x = D[i, j]
            if x == -1:
                continue
            if x < min_v:
                min_v = x
                min_idx = (i, j)
    return min_idx


def delete_rowcolumn(D, i, l):
    """
    Updates a distance matrix D by removing the row and column i
    It actually moves the row / column i to the end of the matrix so that
    there's no creation of a new array (more efficient)
    """
    l = D.shape[0] - l
    e = -l + 1 if -l + 1 != 0 else D.shape[0]

    D[:i, i:-l] = D[:i, i + 1:e]
    D[i:-l, :i] = D[i + 1:e, :i]
    D[i:-l, i:-l] = D[i + 1:e, i + 1:e]
    D[-l, :] = -1
    D[:, -l] = -1
    return D


class Cluster:
    """
    Class representing each cluster formed by the algorithm
    """
    def __init__(self, X, points, indices, M=None):
        assert len(points) == len(indices)
        self.X = X
        self.points = points
        self.indices = indices
        self.F = None
        self.M = M
        self.d = 0

    def merge(self, other):
        self.X.extend(other.X)
        self.points.extend(other.points)
        self.indices.extend(other.indices)
        self.M = None
        self.d = None

    def update_mean(self, pool=None):
        iters = 5 if len(self.points) < 100 else 3 if len(self.points) < 250 else 1
        self.M = km(self.points, len(self.points), 1e-6, iters, pool)  # TODO
        self.update_distance(pool=pool)

    def update_distance(self, pool=None):
        if pool is None:
            self.d = sum([d_geodesic(self.M, Mx) for Mx in self.points])
        else:
            self.d = sum(pool.starmap(d_geodesic, ((self.M, Mx) for Mx in self.points)))

    def __len__(self):
        return len(self.points)


class ACDT:
    """
    The ACDT algorithm for clustering
    """
    def __init__(self, k, l, d, X, minimum_ckpt=100, store_every=0, visualize=False):
        """
        k: number of neighbors during knn.
        l: number of target clusters.
        d: subspace dimensional size
        X: The dataset
        minimum_ckpt: if the number of clusters is < minimum_ckpt the checkpoints are saved
        ckpt_every: The interval between checkpoints (0 no checkpoints)
        """
        self.n = X.shape[0]
        self.X = X
        self.k = k
        self.l = l
        self.d = d
        self.minimum_ckpt = minimum_ckpt
        self.store_every = store_every
        self.visualize = visualize
        self.checkpoints = {'knn': self.k}
        PROCESS = os.cpu_count()
        self.pool = Pool(processes=PROCESS)

        self.knn = NearestNeighbors(n_neighbors=self.k + 1, metric='euclidean').fit(X)
        k_indices = self.knn.kneighbors(X, return_distance=False)[:, 1:]  # Compute k-nearest neighbors indices
        self.E = self.knn.kneighbors_graph(X).astype(np.int)  # This is not used but is the edge map
        self.D = np.ones((self.n, self.n)) * -1

        self.C = []
        self.map_cluster = []  # Maps sample idx to cluster containing it
        for i, x in enumerate(X):
            Nx = X[k_indices[i]]  # Take k neighbors
            N0x = (Nx - x).T  # Translate neighborhood to the origin
            u_N0x, _, _ = scipy.linalg.svd(N0x, full_matrices=False)
            M = u_N0x[:, :d]  # Take d-rank svd
            self.C.append(Cluster([x], [M], [i], M))
            self.map_cluster.append(self.C[-1])

        # Precompute all the distances and populate D
        print('Initializing the distances between clusters')
        pairs = {}
        for Ci in self.C:
            neigh = set(self.knn.kneighbors(Ci.X, return_distance=False)[:, 1:].flatten())
            neigh_c = set([self.map_cluster[n] for n in neigh])
            pairs[Ci] = [(Ci, Cj) if self.C.index(Ci) < self.C.index(Cj) else (Cj, Ci) for Cj in neigh_c if Ci != Cj]
        pairs = set(itertools.chain.from_iterable(pairs.values()))  # Single list of pairs

        distances = self.pool.starmap(d_hat, pairs)
        for ith, (Ci, Cj) in enumerate(pairs):
            i, j = self.C.index(Ci), self.C.index(Cj)
            i, j = min(i, j), max(i, j)  # So that we always populate upper part
            self.D[i, j] = distances[ith]
        print('Computed %s distances' % np.count_nonzero(self.D != -1))

    def fusible(self, Ci, Cj):
        """
        To check if two clusters are fusible we have to see if there's an edge connecting them
        we can iterate on Ci and see if it is connected to any sample in Cj.

        This is not used as is very inefficient
        """
        for i in Ci.indices:
            for j in Cj.indices:
                if self.E[i, j] == 1:
                    return True
        return False

    def fit(self):
        for _ in tqdm(range(self.n, self.l, -1)):
            i, j = argmin_dissimilarity(self.D, len(self.C))
            if i == -1 or j == -1:
                print('No clusters can be merged, probably k is too low, stopping with %s clusters' % len(self.C))
                break
            i, j = min(i, j), max(i, j)
            Ci, Cj = self.C[i], self.C[j]

            # Save neighbors of Cj
            l = self.D.shape[0] - len(self.C)
            e = -l + 1 if -l + 1 != 0 else self.D.shape[0]
            ind_r = np.argwhere(self.D[:j, j] != -1)
            ind_c = np.argwhere(self.D[j, j + 1:e] != -1) + j + 1
            Cj_neigh = [self.C[idx.item()] for idx in itertools.chain(ind_r, ind_c)]

            self.C.remove(Cj)
            Ci.merge(Cj)
            Ci.update_mean(pool=self.pool)

            for s_idx in Cj.indices:
                self.map_cluster[s_idx] = Ci

            # Update distances
            self.D = delete_rowcolumn(self.D, j, len(self.C))  # Delete (offset all to left to skip it) column and row j

            # Now for each connection with Ci update the distance
            neigh = set(self.knn.kneighbors(Ci.X, return_distance=False)[:, 1:].flatten())
            Ci_neigh = set([self.map_cluster[n] for n in neigh])
            index_Ci = self.C.index(Ci)
            pairs = set((Ci, Cj) if index_Ci < self.C.index(Cj) else (Cj, Ci) for Cj in Ci_neigh if Ci != Cj)
            pairs.update((Ci, Cj) if index_Ci < self.C.index(Cj) else (Cj, Ci) for Cj in Cj_neigh if Ci != Cj)

            distances = self.pool.starmap(d_hat, pairs)
            for ith, (Ci, Cj) in enumerate(pairs):
                i, j = self.C.index(Ci), self.C.index(Cj)
                i, j = min(i, j), max(i, j)  # So that we always populate upper part
                self.D[i, j] = distances[ith]

            if self.store_every != 0 and len(self.C) < self.minimum_ckpt and len(self.C) % self.store_every == 0:
                for Ci in self.C:
                    samples = np.array(Ci.X).T
                    mean_pos = np.mean(samples, axis=1, keepdims=True)
                    C0mi = samples - mean_pos
                    u_C0mi, _, _ = scipy.linalg.svd(C0mi, full_matrices=False)
                    Ci.F = u_C0mi[:, :self.d]

                self.checkpoints[len(self.C)] = {'C': copy.deepcopy(self.C)}

                if self.visualize:
                    if self.X.shape[1] == 2:
                        draw_spiral_clusters(self.C, self.k)
                    if self.X.shape[1] == 3:
                        draw_3d_clusters(self.C)

        # Compute final flats
        for Ci in self.C:
            samples = np.array(Ci.X).T
            mean_pos = np.mean(samples, axis=1, keepdims=True)
            C0mi = samples - mean_pos
            u_C0mi, s, _ = scipy.linalg.svd(C0mi, full_matrices=False)
            Ci.F = u_C0mi[:, :self.d]

        def clear_checkpoints(self):
            self.checkpoints = {'knn': self.k}
