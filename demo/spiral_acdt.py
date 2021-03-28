import scipy
import numpy as np
from datasets_util import make_spiral
from karcher_mean import karcher_mean
from sklearn.neighbors import NearestNeighbors
import itertools


class Cluster:
    def __init__(self, points, indices, M=None):
        assert len(points) == len(indices)
        self.points = points
        self.indices = indices
        self.M = M

    def merge(self, other):
        self.points.extend(other.points)
        self.indices.extend(other.indices)
        self.M = None

    def compute_mean(self):
        karcher_mean(self.points, len(self.points))

    def __len__(self):
        return len(self.points)


def argmin_dissimilarity(C, E):
    Ci_min = None
    Cj_min = None
    d_min = np.inf
    for Ci, Cj in itertools.combinations(C, r=2):
        if not fusible(E, Ci, Cj):
            #print(Ci.indices, Cj.indices, 'are NOT fusible')
            continue

        d = d_hat(Ci, Cj)
        #print(Ci.indices, Cj.indices, 'are fusible, with distance', d, d_min)
        if d < d_min:
            Ci_min = Ci
            Cj_min = Cj
            d_min = d
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
    We can use i=1..len(Ci) and j=i..len(Cj) as the relation is symmetric
    """
    for ith, i in enumerate(Ci.indices):
        for j in Cj.indices[ith:]:
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
    assert np.logical_and(sQaTQb >= 0, sQaTQb <= 1 + 1e-4)  # Numerical error
    thetas = np.arccos(np.clip(sQaTQb, a_min=0, a_max=1))
    # print('THETA SQATQB', thetas, sQaTQb)
    # Equivalent to scipy.linalg.subspace_angles
    # assert np.allclose(np.linalg.norm(thetas), np.linalg.norm(scipy.linalg.subspace_angles(ua_smaller, ub_smaller)), atol=1e-4)
    return np.linalg.norm(thetas, ord=2)


# Params
k = 5
l = 10
d = 1

# dataset points
n = 100

X = make_spiral(n=n, normalize=True)

knn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean').fit(X)
_, k_indices = knn.kneighbors(X)  # Compute k-nearest neighbors indices
k_indices = k_indices[:, 1:]  # TODO: is this needed? I don't think so and the graph is altered
E = knn.kneighbors_graph(X)

M = []
for i, x in enumerate(X):
    Nx = X[k_indices[i]]  # Take k neighbors
    N0x = Nx - x  # Translate neighborhood to the origin
    u_N0x, _, _ = np.linalg.svd(N0x, full_matrices=False)
    M.append(u_N0x[:, :d])  # Take d-rank svd
    # u_N0x, s, vh = np.linalg.svd(N0x, full_matrices=False)  # Check reconstruction
    # print(np.linalg.norm(N0x - np.dot(u_N0x[:, :d] * s[:d], vh[:d, :])))

# Clustering
n = len(X)
lam = 0
C = [Cluster([M[i]], [i], M[i]) for i, x in enumerate(X)]
while lam < n - l:
    Ci, Cj = argmin_dissimilarity(C, E)
    C.remove(Ci)
    C.remove(Cj)
    C.append()
    print(Ci.indices, Cj.indices)
    assert False
    lam += 1
