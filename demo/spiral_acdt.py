import time
import numpy as np
import itertools
import networkx as nx
import sklearn.datasets as datasets
import os
from multiprocessing import Pool
from datasets_util import make_spiral, make_spiral2
from draw_utils import draw_spiral_clusters, draw_3d_clusters
import karcher_mean
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

    def merge(self, other):
        self.X.extend(other.X)
        self.points.extend(other.points)
        self.indices.extend(other.indices)
        self.M = None

    def update_mean(self):
        self.M = km(self.points, len(self.points), 1e-8, 1000)

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

    # Candy multiprocessing for parallel logq_map
    combs = list(itertools.combinations(C, r=2))
    with Pool(processes=PROCESS) as pool:
        d_hats = pool.starmap(d_clusters, [(Ci, Cj, E) for Ci, Cj in combs])
    min_idx = np.argmin(d_hats)
    Ci_min = combs[min_idx][0]
    Cj_min = combs[min_idx][1]
    return Ci_min, Cj_min


def d_hat(Ci, Cj):
    d_mean = d_geodesic(Ci.M, Cj.M)
    d = (len(Ci) + len(Cj)) * (d_mean ** 2)
    d += 2 * d_mean * (sum([d_geodesic(Ci.M, Mx) for Mx in Ci.points]) + sum([d_geodesic(Cj.M, Mx) for Mx in Cj.points]))
    return d


def fusible(E, Ci, Cj):
    """
    To check if two clusters are fusible we have to see if there's an edge connecting them
    we can iterate on Ci and see if it is connected to any sample in Cj.
    """
    for ith, i in enumerate(Ci.indices):
        for j in Cj.indices:
            if E[i, j] == 1:
                return True
    return False


def d_geodesic(x, y):
    # Reference: http://dx.doi.org/10.1137%2FS1064827500377332 Page 3
    # This is unstable numerically but the error should be negligible here
    ua, sa, vha = np.linalg.svd(x)
    ub, sb, vhb = np.linalg.svd(y)
    ua_smaller = ua[:, np.where(~np.isclose(sa, 0))].squeeze(1)
    ub_smaller = ub[:, np.where(~np.isclose(sb, 0))].squeeze(1)
    QaTQb = np.dot(ua_smaller.T, ub_smaller)
    uQaTQb, sQaTQb, vhQaTQb = np.linalg.svd(QaTQb)
    assert np.logical_and(np.all(sQaTQb >= 0), np.all(sQaTQb <= 1 + 1e-4))  # Numerical error
    thetas = np.arccos(np.clip(sQaTQb, a_min=0, a_max=1))
    # print('THETA SQATQB', thetas, sQaTQb)
    # Equivalent to scipy.linalg.subspace_angles
    # assert np.allclose(np.linalg.norm(thetas), np.linalg.norm(scipy.linalg.subspace_angles(ua_smaller, ub_smaller)), atol=1e-4)
    return np.linalg.norm(thetas, ord=2)


# Params
k = 5  # 3d swiss roll
# k = 2  # 2d spiral
l = 12
d = 2

# dataset points
n = 1000
# X = make_spiral(n=n, normalize=True)
X, _ = datasets.make_swiss_roll(n)

knn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean').fit(X)
_, k_indices = knn.kneighbors(X)  # Compute k-nearest neighbors indices
k_indices = k_indices[:, 1:]  # TODO ?
E = knn.kneighbors_graph(X)
G = nx.from_scipy_sparse_matrix(E)

M = []
for i, x in enumerate(X):
    Nx = X[k_indices[i]]  # Take k neighbors
    N0x = Nx - x  # Translate neighborhood to the origin
    u_N0x, _, _ = np.linalg.svd(N0x, full_matrices=False)
    M.append(u_N0x[:, :d])  # Take d-rank svd
    # u_N0x, s, vh = np.linalg.svd(N0x, full_matrices=False)  # Check reconstruction
    # print(np.linalg.norm(N0x - np.dot(u_N0x[:, :d] * s[:d], vh[:d, :])))

# Clustering
t = time.time()
n = len(X)
lam = 0
C = [Cluster([x], [M[i]], [i], M[i]) for i, x in enumerate(X)]
while lam < n - l:
    Ci, Cj = argmin_dissimilarity(C, E)
    print(Ci.indices, Cj.indices)
    C.remove(Cj)
    Ci.merge(Cj)
    Ci.update_mean()
    lam += 1
    print('Total Clusters: %s' % len(C))
    print('Time for this merge: %s' % (time.time() - t))
    t = time.time()

for Ci in C:
    samples = np.array(Ci.X).T
    mean_pos = np.mean(samples, axis=1, keepdims=True)
    C0mi = samples - mean_pos
    u_C0mi, s, _ = np.linalg.svd(C0mi, full_matrices=False)
    Ci.F = u_C0mi[:, :d]

if X.shape[1] == 2:
    draw_spiral_clusters(X, C, G)
if X.shape[1] == 3:
    draw_3d_clusters(X, C)
