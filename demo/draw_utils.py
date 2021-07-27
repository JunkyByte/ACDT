import networkx as nx
import numpy as np
import random
import matplotlib.cm
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


def draw_spiral_clusters(C, k):
    X = sum([Ci.X for Ci in C], [])
    colors = [np.array(random.sample(range(0, 256), 3)) / 256 for i in range(len(C))]
    colors_list = sum([[colors[i]] * len(C[i]) for i in range(len(C))], [])
    width = 0.05 if len(X) > 200 else 0.25
    knn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean').fit(X)
    E = knn.kneighbors_graph(X).astype(np.int)
    G = nx.from_scipy_sparse_matrix(E, create_using=nx.MultiGraph)
    nx.draw(G, pos=X, node_size=40, node_color=colors_list, width=width)

    for i, Ci in enumerate(C):
        if len(Ci) == 1:  # Skip singleton clusters
            continue

        flat = Ci.F
        samples = np.array(Ci.X).T
        p = np.mean(samples, axis=1, keepdims=True)

        length_x = max(np.abs(np.max(Ci.X, axis=0)[0] - np.min(Ci.X, axis=0)[0]), 0.1)
        length_y = max(np.abs(np.max(Ci.X, axis=0)[1] - np.min(Ci.X, axis=0)[1]), 0.1)
        pts = [-10, 10]
        xs = np.clip(pts * flat[0], -length_x / 2, length_x / 2)
        ys = np.clip(pts * flat[1], -length_y / 2, length_y / 2)
        plt.autoscale(False)

        plt.plot(xs + p[0], ys + p[1], '-', lw=2.25, color=colors[i])
        plt.scatter(p[0], p[1], color=colors[i], edgecolors='black')
    plt.show()


def draw_3d_clusters(C):
    X = np.array(sum([Ci.X for Ci in C], []))
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    colors = [np.random.rand(3, ) for i in range(len(C))]
    colors_list = sum([[c] * len(C[i]) for i, c in enumerate(colors)], [])
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors_list)

    for i, Ci in enumerate(C):
        if len(Ci) == 1:  # Skip singleton clusters
            continue

        flat = Ci.F
        samples = np.array(Ci.X).T
        p = np.mean(samples, axis=1, keepdims=True)

        p0 = np.dot(flat, [1, 0])
        p1 = np.dot(flat, [0, 1])
        p2 = np.dot(flat, [1, 1])

        x0, y0, z0 = p0
        x1, y1, z1 = p1
        x2, y2, z2 = p2

        ux, uy, uz = u = [x1-x0, y1-y0, z1-z0]
        vx, vy, vz = v = [x2-x0, y2-y0, z2-z0]

        u_cross_v = [uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx]

        point  = np.array(p0)
        normal = np.array(u_cross_v)

        d = -point.dot(normal)

        xx, yy = np.meshgrid(range(-10000, 10000, 5000), range(-10000, 10000, 5000))
        length_x = max(np.abs(np.max(Ci.X, axis=0)[0] - np.min(Ci.X, axis=0)[0]), 0.1)
        length_y = max(np.abs(np.max(Ci.X, axis=0)[1] - np.min(Ci.X, axis=0)[1]), 0.1)
        length_z = max(np.abs(np.max(Ci.X, axis=0)[2] - np.min(Ci.X, axis=0)[2]), 0.1)
        xx = np.clip(xx, -length_x / 2, length_x / 2)
        yy = np.clip(yy, -length_y / 2, length_y / 2)
        z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
        mask = np.where((z - p[2] > -length_z / 2) & (z < length_z / 2))[0]
        xx = xx[mask]
        yy = yy[mask]
        z = z[mask]

        plt.autoscale(False)
        ax.plot_surface(xx + p[0], yy + p[1], z + p[2], color=colors[i], alpha=0.25, shade=False)

    plt.show()
