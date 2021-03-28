import numpy as np


def make_spiral(n=100, normalize=False):
    ls = np.linspace(0, 2 * np.pi, n)
    v = 1
    w = np.pi
    c = 0
    xs = (v * ls + c) * np.cos(w * ls)
    ys = (v * ls + c) * np.sin(w * ls)
    out = np.concatenate([xs[..., None], ys[..., None]], axis=1)
    if normalize:
        out -= np.mean(out)
        out /= np.std(out)
    return out
