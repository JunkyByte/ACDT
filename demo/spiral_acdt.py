import time
import numpy as np
import itertools
import networkx as nx
import sklearn.datasets as datasets
import karcher_mean
import scipy.linalg
import os
from datasets_util import make_spiral, make_spiral2, make_spiral3
from draw_utils import draw_spiral_clusters, draw_3d_clusters
from multiprocessing import Pool
from karcher_mean import karcher_mean as km
from sklearn.neighbors import NearestNeighbors
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
        self.M = km(self.points, len(self.points), 1e-6, 1000)
        self.update_distance()

    def update_distance(self):
        self.d = sum([d_geodesic(self.M, Mx) for Mx in self.points])

    def __len__(self):
        return len(self.points)


def d_clusters(Ci, Cj, E):
    if not fusible(E, Ci, Cj):
        # print(Ci.indices, Cj.indices, 'are NOT fusible')
        return np.inf

    d = d_hat(Ci, Cj)
    # print(Ci.indices, Cj.indices, 'are fusible, with distance', d, d_min)
    return d


def argmin_dissimilarity(C, E):
    Ci_min = None
    Cj_min = None
    combs = list(itertools.combinations(C, r=2))
    d_hats = pool.starmap(d_clusters, [(Ci, Cj, E) for Ci, Cj in combs])
    min_idx = np.argmin(d_hats)
    Ci_min = combs[min_idx][0]
    Cj_min = combs[min_idx][1]
    return Ci_min, Cj_min


def d_hat(Ci, Cj):
    d_mean = d_geodesic(Ci.M, Cj.M)
    d = (len(Ci) + len(Cj)) * (d_mean ** 2) + 2 * d_mean * (Ci.d + Cj.d)
    return d


def fusible(E, Ci, Cj):
    """
    To check if two clusters are fusible we have to see if there's an edge connecting them
    we can iterate on Ci and see if it is connected to any sample in Cj.
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
    ua_smaller = ua[:, :sa.shape[0]]
    ub_smaller = ub[:, :sa.shape[0]]
    QaTQb = np.dot(ua_smaller.T, ub_smaller)
    uQaTQb, sQaTQb, vhQaTQb = scipy.linalg.svd(QaTQb)
    sQaTQb.clip(0, 1, out=sQaTQb)
    thetas = np.arccos(sQaTQb)
    return np.linalg.norm(thetas, ord=2)


pool = Pool(processes=PROCESS)

# Params
k = 5
# k = 5
# l = 12
l = 15
d = 1

# dataset points
# n = 2000
n = 200
X = make_spiral2(n=n, normalize=True)
# X = make_spiral3(n=n, normalize=True)
# X, _ = datasets.make_swiss_roll(n)

knn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean').fit(X)
_, k_indices = knn.kneighbors(X)  # Compute k-nearest neighbors indices
k_indices = k_indices[:, 1:]  # TODO ?
E = knn.kneighbors_graph(X).astype(np.int)
G = nx.from_scipy_sparse_matrix(E)

M = []
for i, x in enumerate(X):
    Nx = X[k_indices[i]]  # Take k neighbors
    N0x = Nx - x  # Translate neighborhood to the origin
    u_N0x, _, _ = scipy.linalg.svd(N0x, full_matrices=False)
    M.append(u_N0x[:, :d])  # Take d-rank svd
    # u_N0x, s, vh = np.linalg.svd(N0x, full_matrices=False)  # Check reconstruction
    # print(np.linalg.norm(N0x - np.dot(u_N0x[:, :d] * s[:d], vh[:d, :])))

# Clustering
total = time.time()
n = len(X)
lam = 0
C = [Cluster([x], [M[i]], [i], M[i]) for i, x in enumerate(X)]
while lam < n - l:
    t = time.time()
    Ci, Cj = argmin_dissimilarity(C, E)
    print('Merged: %s with %s' % (Ci.indices, Cj.indices))
    C.remove(Cj)
    Ci.merge(Cj)
    Ci.update_mean()
    lam += 1
    print('Total Clusters: %s' % len(C))
    print('Time for this merge: %s' % (time.time() - t))

# Close multiprocessing pools
pool.close()
karcher_mean.pool.close()

for Ci in C:
    samples = np.array(Ci.X).T
    mean_pos = np.mean(samples, axis=1, keepdims=True)
    C0mi = samples - mean_pos
    u_C0mi, s, _ = scipy.linalg.svd(C0mi, full_matrices=False)
    Ci.F = u_C0mi[:, :d]

print(time.time() - total)

if X.shape[1] == 2:
    draw_spiral_clusters(C, G)
if X.shape[1] == 3:
    draw_3d_clusters(C)
