import time
import numpy as np
import itertools
import networkx as nx
import sklearn.datasets as datasets
import karcher_mean
import scipy.linalg
import pickle
import os
from datasets_util import make_spiral, make_2_spiral
from draw_utils import draw_spiral_clusters, draw_3d_clusters
from multiprocessing import Pool
from karcher_mean import karcher_mean as km
from sklearn.neighbors import NearestNeighbors
from numba import njit
PROCESS = os.cpu_count()


class Cluster:
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

    def update_mean(self):
        self.M = km(self.points, len(self.points), 1e-6, 50)  # TODO
        self.update_distance()

    def update_distance(self):
        self.d = sum([d_geodesic(self.M, Mx) for Mx in self.points])

    def __len__(self):
        return len(self.points)


@njit(cache=True)
def argmin_dissimilarity(D):
    n, m = D.shape
    min_v = np.inf
    min_idx = (-1, -1)
    for i in range(n):
        for j in range(i, m):
            x = D[i, j]
            if x == -1:
                continue
            if x < min_v:
                min_v = x
                min_idx = (i, j)
    return min_idx


def d_hat(Ci, Cj):
    d_mean = d_geodesic(Ci.M, Cj.M)
    d = (len(Ci) + len(Cj)) * (d_mean ** 2) + 2 * d_mean * (Ci.d + Cj.d)
    print(d, len(Ci), len(Cj), d_mean, Ci.d, Cj.d)
    return d


def fusible(E, Ci, Cj):
    """
    To check if two clusters are fusible we have to see if there's an edge connecting them
    we can iterate on Ci and see if it is connected to any sample in Cj.

    This is not used as is very inefficient
    """
    for i in Ci.indices:
        for j in Cj.indices:
            if E[i, j] == 1:
                return True
    return False


def d_geodesic(x, y):
    # Reference: http://dx.doi.org/10.1137%2FS1064827500377332 Page 3
    # This is unstable numerically but the error should be negligible here
    ua, sa, vha = scipy.linalg.svd(x)
    ub, sb, vhb = scipy.linalg.svd(y)
    QaTQb = np.dot(ua.T, ub)
    uQaTQb, sQaTQb, vhQaTQb = scipy.linalg.svd(QaTQb)
    sQaTQb.clip(0, 1, out=sQaTQb)
    thetas = np.arccos(sQaTQb)
    return np.linalg.norm(thetas, ord=2)


pool = Pool(processes=PROCESS)

# Params
k = 5
l = 15
d = 1

# dataset points
n = 300
# X = make_spiral(n=n, normalize=True)
X = make_2_spiral(n=n, normalize=True)
# X, _ = datasets.make_swiss_roll(n)

knn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean').fit(X)
k_indices = knn.kneighbors(X, return_distance=False)[:, 1:]  # Compute k-nearest neighbors indices
E = knn.kneighbors_graph(X).astype(np.int) # These are not used
G = nx.from_scipy_sparse_matrix(E.copy(), create_using=nx.MultiGraph)
D = np.ones((n, n)) * -1

C = []
map_cluster = []  # Maps sample idx to cluster containing it
for i, x in enumerate(X):
    Nx = X[k_indices[i]]  # Take k neighbors
    N0x = Nx - x  # Translate neighborhood to the origin

    N0x = N0x.T

    u_N0x, _, _ = scipy.linalg.svd(N0x, full_matrices=False)
    M = u_N0x[:, :d]  # Take d-rank svd
    # u_N0x, s, vh = np.linalg.svd(N0x, full_matrices=False)  # Check reconstruction
    # print(np.linalg.norm(N0x - np.dot(u_N0x[:, :d] * s[:d], vh[:d, :])))
    C.append(Cluster([x], [M], [i], M))
    map_cluster.append(C[-1])

# Precompute all the distances and populate D
pairs = {}
for Ci in C:
    neigh = set(knn.kneighbors(Ci.X, return_distance=False)[:, 1:].flatten())
    neigh_c = set([map_cluster[n] for n in neigh])
    pairs[Ci] = [(Ci, Cj) for Cj in neigh_c if Cj not in pairs.keys() and Ci != Cj]  # This skips already merged clusters
pairs = list(itertools.chain.from_iterable(pairs.values()))  # Single list of pairs

for Ci, Cj in pairs:
    i, j = C.index(Ci), C.index(Cj)
    i, j = min(i, j), max(i, j)  # So that we always populate upper part
    assert j > i  # TODO

    D[i, j] = d_hat(Ci, Cj)

# Clustering
total = time.time()
n = len(X)
lam = 0
while lam < n - l:
    t = time.time()
    i, j = argmin_dissimilarity(D)
    i, j = min(i, j), max(i, j)
    Ci, Cj = C[i], C[j]

    # print('Merged: %s with %s' % (Ci.indices, Cj.indices))
    C.remove(Cj)
    Ci.merge(Cj)
    Ci.update_mean()

    for s_idx in Cj.indices:
        map_cluster[s_idx] = Ci

    # np.set_printoptions(precision=4, suppress=True)
    # print(D)

    # Update distances
    D = np.delete(np.delete(D, j, axis=1), j, axis=0)  # Delete column and row j
    print('Deleted row and column j?', j)

    # Now for each connection of Ci update the distance
    neigh = set(knn.kneighbors(Ci.X, return_distance=False)[:, 1:].flatten())
    neigh_c = set([map_cluster[n] for n in neigh])
    pairs = [(Ci, Cj) for Cj in neigh_c if Ci != Cj]
    print('Ci has to update', len(pairs))

    for Ci, Cj in pairs:
        i, j = C.index(Ci), C.index(Cj)
        i, j = min(i, j), max(i, j)  # So that we always populate upper part
        D[i, j] = d_hat(Ci, Cj)

    lam += 1
    print('Total Clusters: %s' % len(C))
    print('Time for this merge: %s' % (time.time() - t))

    # if len(C) < 100 and lam % 10 == 0:
    #     for Ci in C:
    #         samples = np.array(Ci.X).T
    #         mean_pos = np.mean(samples, axis=1, keepdims=True)
    #         C0mi = samples - mean_pos
    #         u_C0mi, s, _ = scipy.linalg.svd(C0mi, full_matrices=False)
    #         Ci.F = u_C0mi[:, :d]
    #     if X.shape[1] == 2:
    #         draw_spiral_clusters(C, k)
    #     if X.shape[1] == 3:
    #         draw_3d_clusters(C)

# Close multiprocessing pools
# pool.close()
# karcher_mean.pool.close()

for Ci in C:
    samples = np.array(Ci.X).T
    mean_pos = np.mean(samples, axis=1, keepdims=True)
    C0mi = samples - mean_pos
    u_C0mi, s, _ = scipy.linalg.svd(C0mi, full_matrices=False)
    Ci.F = u_C0mi[:, :d]

print(time.time() - total)

# Save the data for further visualization
# data = {
#     'C': C,
#     'knn': knn
# }
# 
# PATH = './saved/'
# os.makedirs(PATH, exist_ok=True)
# with open(os.path.join(PATH, 'ckpt.pickle'), 'wb') as f:
#     pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
# 
if X.shape[1] == 2:
    draw_spiral_clusters(C, k)
if X.shape[1] == 3:
    draw_3d_clusters(C)
