import networkx as nx
import numpy as np
import random
import matplotlib.cm
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
colors = [np.array(random.sample(range(0, 256), 3)) / 256 for i in range(5000)]


def draw_spiral_clusters(C, k, draw_flats=True):
    X = sum([Ci.X for Ci in C], [])
    colors_list = sum([[colors[C[i].idx]] * len(C[i]) for i in range(len(C))], [])
    width = 0.01 if len(X) > 200 else 0.25
    knn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean').fit(X)
    E = knn.kneighbors_graph(X).astype(np.int)
    G = nx.from_scipy_sparse_matrix(E, create_using=nx.MultiGraph)
    nx.draw(G, pos=X, node_size=40, node_color=colors_list, width=width)

    for i, Ci in enumerate(C):
        if len(Ci) == 1:  # Skip singleton clusters
            continue

        if draw_flats:
            flat = Ci.F
            samples = np.array(Ci.X).T
            p = np.mean(samples, axis=1, keepdims=True)

            length_x = max(np.abs(np.max(Ci.X, axis=0)[0] - np.min(Ci.X, axis=0)[0]), 0.1)
            length_y = max(np.abs(np.max(Ci.X, axis=0)[1] - np.min(Ci.X, axis=0)[1]), 0.1)
            pts = [-100000, 100000]
            xs = np.clip(pts * flat[0], -length_x / 2, length_x / 2)
            ys = np.clip(pts * flat[1], -length_y / 2, length_y / 2)
            plt.autoscale(False)

            plt.plot(xs + p[0], ys + p[1], 'k-', lw=2.25, color=colors[i])
            plt.scatter(p[0], p[1], color=colors[i], edgecolors='black')
    plt.show()


def draw_3d_clusters(C):
    X = np.array(sum([Ci.X for Ci in C], []))
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    colors_list = sum([[colors[C[i].idx]] * len(C[i]) for i in range(len(C))], [])
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors_list)
    plt.show()
