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
PROCESS = os.cpu_count()


class Cluster:
    def __init__(self, X, N, points, indices, M=None):
        assert len(points) == len(indices)
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
        self.M = km(self.points, len(self.points), 1e-6, 50)  # TODO
        self.update_distance()
        self.update_svd()

    def update_distance(self):
        self.d = sum([d_geodesic(self.M, Mx) for Mx in self.points])

    def update_svd(self):
        ua, sa, vha = svd(self.M)
        self.Us = ua[:, :sa.shape[0]]

    def __len__(self):
        return len(self.points)


def argmin_dissimilarity(C, knn):
    t = time.time()
    pairs = {}
    for Ci in C:
        neigh = set(Ci.N.flatten())
        neigh_c = [map_cluster[n] for n in neigh]
        pairs[Ci] = [(Ci, Cj) for Cj in neigh_c if Cj not in pairs.keys() and Ci != Cj]  # This skips already merged clusters
    pairs = sum(pairs.values(), [])  # Single list of pairs
    print('Checking %s valid mergings' % len(pairs))
    print('Time to generate valid mergings:', time.time() - t)

    t = time.time()
    min_idx = np.argmin(pool.starmap(d_hat, pairs))
    print('Time to argmin distance:', time.time() - t)
    return pairs[min_idx]


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


def d_geodesic(ua, ub):
    # Reference: http://dx.doi.org/10.1137%2FS1064827500377332 Page 3
    # This is unstable numerically but the error should be negligible here
    # inputs are reduced left matrix of svd U for x and y
    QaTQb = np.dot(ua.T, ub)
    uQaTQb, sQaTQb, vhQaTQb = svd(QaTQb)
    sQaTQb.clip(0, 1, out=sQaTQb)
    thetas = np.arccos(sQaTQb)
    return np.linalg.norm(thetas, ord=2)


pool = Pool(processes=PROCESS)

# Params
k = 4
l = 40
d = 1

# dataset points
n = 200
X = make_spiral(n=n, normalize=True)
# X = make_2_spiral(n=n, normalize=True)
# X, _ = datasets.make_swiss_roll(n)

knn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean').fit(X)
k_indices = knn.kneighbors(X, return_distance=False)[:, 1:]  # Compute k-nearest neighbors indices
E = knn.kneighbors_graph(X).astype(np.int) # These are not used
G = nx.from_scipy_sparse_matrix(E.copy(), create_using=nx.MultiGraph)

C = []
map_cluster = []  # Maps sample idx to cluster containing it
for i, x in enumerate(X):
    Nx = X[k_indices[i]]  # Take k neighbors
    N0x = Nx - x  # Translate neighborhood to the origin
    u_N0x, _, _ = svd(N0x, full_matrices=False)
    M = u_N0x[:, :d]  # Take d-rank svd
    # u_N0x, s, vh = np.linalg.svd(N0x, full_matrices=False)  # Check reconstruction
    # print(np.linalg.norm(N0x - np.dot(u_N0x[:, :d] * s[:d], vh[:d, :])))
    C.append(Cluster([x], knn.kneighbors([x], return_distance=False)[:, 1:], [M], [i], M))
    map_cluster.append(C[-1])

# Clustering
total = time.time()
n = len(X)
lam = 0
while lam < n - l:
    t = time.time()
    Ci, Cj = argmin_dissimilarity(C, knn)
    print('Merged: %s with %s' % (Ci.indices, Cj.indices))
    C.remove(Cj)
    Ci.merge(Cj)
    Ci.update_mean()
    Ci.N = knn.kneighbors(Ci.X, return_distance=False)[:, 1:]

    for s_idx in Cj.indices:
        map_cluster[s_idx] = Ci

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
    u_C0mi, s, _ = svd(C0mi, full_matrices=False)
    Ci.F = u_C0mi[:, :d]

print(time.time() - total)

# Save the data for further visualization
# data = {
#     'C': C,
#     'knn': knn
# }

# PATH = './saved/'
# os.makedirs(PATH, exist_ok=True)
# with open(os.path.join(PATH, 'ckpt.pickle'), 'wb') as f:
#     pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

if X.shape[1] == 2:
    draw_spiral_clusters(C, k)
if X.shape[1] == 3:
    draw_3d_clusters(C)
