import networkx as nx
import numpy as np
import matplotlib.cm
import matplotlib.pyplot as plt
cmap = matplotlib.cm.get_cmap('rainbow')


def draw_spiral_clusters(C, G):
    X = sum([Ci.X for Ci in C], [])
    colors = [np.random.rand(3, ) for i in range(len(C))]
    colors_list = sum([[colors[i]] * len(C[i]) for i in range(len(C))], [])
    nx.draw(G, pos=X, node_size=40, node_color=colors_list)

    for i, Ci in enumerate(C):
        if len(Ci) == 1:  # Skip singleton clusters
            continue

        flat = Ci.F
        samples = np.array(Ci.X).T
        p = np.mean(samples, axis=1, keepdims=True)

        length_x = max(np.abs(np.max(samples[0, :]) - np.min(samples[0, :])), 0.1)
        length_y = max(np.abs(np.max(samples[1, :]) - np.min(samples[1, :])), 0.1)
        length = max(length_x, length_y)
        pts = np.linspace(-length * 3 / 4, length * 3 / 4, num=2)
        xs = pts * flat[0]
        ys = pts * flat[1]

        plt.autoscale(False)

        plt.plot(xs + p[0], ys + p[1], 'k-', lw=2.25, color=colors[i])
        plt.scatter(p[0], p[1], color=colors[i], edgecolors='black')
    plt.show()


def draw_3d_clusters(C):
    X = np.array(sum([Ci.X for Ci in C], []))
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    colors = [np.random.rand(3, ) for i in range(len(C))]
    colors_list = sum([[c] * len(C[i]) for i, c in enumerate(colors)], [])
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors_list)
    plt.show()
