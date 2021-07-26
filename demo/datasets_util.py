import math
import numpy as np
from numpy import pi

def make_spiral_old(n=100, normalize=False):
    ls = np.linspace(0, 4 * np.pi, n)
    v = 1
    w = 1
    c = 0
    xs = (v * ls + c) * np.cos(w * ls)
    ys = (v * ls + c) * np.sin(w * ls)
    out = np.concatenate([xs[..., None], ys[..., None]], axis=1)
    if normalize:
        out -= np.mean(out)
        out /= np.std(out)
    return out


def make_spiral(n=100, normalize=False):
    theta = np.radians(np.linspace(90, 360 * 4, n))
    theta *= np.geomspace(1, 2.4, n)[::-1]
    r = theta ** 2
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.concatenate([x[..., None], y[..., None]], axis=1)


def make_2_spiral(n=100, normalize=False):
    n = n // 2
    theta = np.sqrt(np.random.rand(n))*2*pi

    r_a = 2*theta + pi
    data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
    x_a = data_a + np.random.randn(n,2)

    r_b = -2*theta - pi
    data_b = np.array([np.cos(theta)*r_b, np.sin(theta)*r_b]).T
    x_b = data_b + np.random.randn(n,2)

    return np.append(x_a, x_b, axis=0)
