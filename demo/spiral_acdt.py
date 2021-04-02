import time
import numpy as np
import itertools
import networkx as nx
import sklearn.datasets as datasets
import karcher_mean
from scipy.linalg import svd
import pickle
import os
from datasets_util import make_spiral, make_2_spiral
from draw_utils import draw_spiral_clusters, draw_3d_clusters
from multiprocessing import Pool
from karcher_mean import karcher_mean as km
from sklearn.neighbors import NearestNeighbors
from itertools import chain
from numba import extending, njit
PROCESS = os.cpu_count()


@extending.overload(np.clip)
def np_clip(a, a_min, a_max, out=None):
    def np_clip_impl(a, a_min, a_max, out=None):
        if out is None:
            out = np.empty_like(a)
        for i in range(len(a)):
            if a[i] < a_min:
                out[i] = a_min
            elif a[i] > a_max:
                out[i] = a_max
            else:
                out[i] = a[i]
        return out
    return np_clip_impl


class Cluster:
    def __init__(self, idx, X, N, points, indices, M=None):
        assert len(points) == len(indices)
        self.idx = idx
        self.X = X
        self.points = points
        self.indices = indices
        self.F = None
        self.M = M
        self.N = N
        self.update_svd()
        self.d = 0

    def merge(self, other):
        self.X.extend(other.X)
        self.points.extend(other.points)
        self.indices.extend(other.indices)
        self.M = None
        self.d = None

    def update_mean(self):
        self.M = km(self.points, len(self.points), 1e-4, 50)  # TODO Inspect this for correct maxit
        self.update_distance()
        self.update_svd()

    def update_distance(self):
        self.d = sum([d_geodesic(self.M, Mx) for Mx in self.points])

    def update_svd(self):
        ua, sa, _ = svd(self.M, check_finite=False)
        self.Us = ua[:, :sa.shape[0]]

    def __len__(self):
        return len(self.points)


def argmin_dissimilarity(C, d):
    t = time.time()
    pairs = {}
    for Ci in C:  # This is the main bottleneck as everything is precomputed
        neigh = Ci.N
        neigh_c = [map_cluster[n] for n in neigh]
        # Compute only pairs that are connected and which distance has to be updated
        pairs[Ci] = [(Ci, Cj) for Cj in neigh_c if d[Ci.idx, Cj.idx] == -1 and Cj not in pairs.keys() and Ci != Cj]
    pairs = list(chain.from_iterable(pairs.values()))  # Single list of pairs
    print('Checking %s valid mergings' % len(pairs))
    print('Time to generate valid mergings:', time.time() - t)

    # t = time.time()
    for i, d_new in enumerate(pool.starmap(d_hat, pairs)):
        Ci, Cj = pairs[i]
        if Ci.idx < Cj.idx:
            i = Ci.idx
            j = Cj.idx
        else:
            i = Cj.idx
            j = Ci.idx
        d[i, j] = d_new

    t2 = time.time()
    i, j = argmin(d)
    print('time for argmin', time.time() - t2)
    # print(i, j)
    Ci = next(Ci for Ci in C if Ci.idx == i)
    Cj = next(Cj for Cj in C if Cj.idx == j)
    print('Time to function armin_diss:', time.time() - t)
    return Ci, Cj, d


@njit(cache=True)
def argmin(A):
    n, m = A.shape
    min_v = np.inf
    min_idx = (-1, -1)
    for i in range(n):
        for j in range(i, m):
            x = A[i, j]
            if x == -1:
                continue
            if x < min_v:
                min_v = x
                min_idx = (i, j)
    return min_idx


def d_hat(Ci, Cj):
    d_mean = d_geodesic(Ci.Us, Cj.Us)
    d = (len(Ci) + len(Cj)) * (d_mean ** 2) + 2 * d_mean * (Ci.d + Cj.d)
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


@njit(cache=True)
def d_geodesic(ua, ub):
    # Reference: http://dx.doi.org/10.1137%2FS1064827500377332 Page 3
    # This is unstable numerically but the error should be negligible here
    # inputs are reduced left matrix of svd U for x and y
    QaTQb = np.dot(ua.T, ub)
    a, sQaTQb, b = np.linalg.svd(QaTQb)
    np.clip(sQaTQb, 0, 1, out=sQaTQb)
    thetas = np.arccos(sQaTQb)
    return np.linalg.norm(thetas, ord=2)


pool = Pool(processes=PROCESS)

# Params
k = 15
l = 60
d = 2

# dataset points
n = 5000
# X = make_spiral(n=n, normalize=True)
# X = make_2_spiral(n=n, normalize=True)
X, _ = datasets.make_swiss_roll(n)

knn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean').fit(X)
k_indices = knn.kneighbors(X, return_distance=False)[:, 1:]  # Compute k-nearest neighbors indices
E = knn.kneighbors_graph(X).astype(np.int) # These are not used
G = nx.from_scipy_sparse_matrix(E.copy(), create_using=nx.MultiGraph)
distances = np.ones((n, n), dtype=np.float32) * -1  # This will be a triangular matrix, I'm only using upper half

C = []
map_cluster = []  # Maps sample idx to cluster containing it
for i, x in enumerate(X):
    Nx = X[k_indices[i]]  # Take k neighbors
    N0x = Nx - x  # Translate neighborhood to the origin
    u_N0x, _, _ = svd(N0x, full_matrices=False, overwrite_a=True, check_finite=False)
    M = u_N0x[:, :d]  # Take d-rank svd
    # u_N0x, s, vh = np.linalg.svd(N0x, full_matrices=False)  # Check reconstruction
    # print(np.linalg.norm(N0x - np.dot(u_N0x[:, :d] * s[:d], vh[:d, :])))
    C.append(Cluster(i, [x], set(knn.kneighbors([x], return_distance=False)[:, 1:].flatten()), [M], [i], M))
    map_cluster.append(C[-1])


# Useful for benchmark to precompile the numba execution
# pool.starmap(d_hat, [(C[0], C[0]) for i in range(PROCESS)])
# km(C[0].points, 1, 1e-6, 50)
########################################################

# Clustering
total = time.time()
n = len(X)
lam = 0
while lam < n - l:
    t = time.time()
    Ci, Cj, distances = argmin_dissimilarity(C, distances)  # This returns the updated distances
    C.remove(Cj)
    Ci.merge(Cj)
    Ci.update_mean()
    Ci.N = set(knn.kneighbors(Ci.X, return_distance=False)[:, 1:].flatten())
    distances[Cj.idx, :] = -1
    distances[:, Cj.idx] = -1
    distances[Ci.idx, :] = -1
    distances[:, Ci.idx] = -1

    for s_idx in Cj.indices:
        map_cluster[s_idx] = Ci

    lam += 1
    print('Clusters: %s/%s Merging took: %s' % (len(C), l, time.time() - t))

# Close multiprocessing pools
pool.close()
karcher_mean.pool.close()

for Ci in C:
    samples = np.array(Ci.X).T
    mean_pos = np.mean(samples, axis=1, keepdims=True)
    C0mi = samples - mean_pos
    u_C0mi, s, _ = svd(C0mi, full_matrices=False)
    Ci.F = u_C0mi[:, :d]

print(time.time() - total)

# Save the data for further visualization
data = {
    'C': C,
    'knn': knn
}

PATH = './saved/'
os.makedirs(PATH, exist_ok=True)
with open(os.path.join(PATH, 'ckpt.pickle'), 'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

if X.shape[1] == 2:
    draw_spiral_clusters(C, k)
if X.shape[1] == 3:
    draw_3d_clusters(C)
