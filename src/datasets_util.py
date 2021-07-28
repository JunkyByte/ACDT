import os
import math
import cv2
import numpy as np
from sklearn.feature_extraction import image
from numpy import pi


def load_bsds(path, n=10000):
    images = [cv2.imread(os.path.join(path, f)) for f in os.listdir(path)]
    # images = [cv2.resize(im, (im.shape[1] // 4, im.shape[0] // 4)) for im in images]
    images = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in images]
    images = np.array([image.extract_patches_2d(im, (8, 8), max_patches=n // len(images) + 1) for im in images])
    # for img in images:
    #     cv2.imshow('', img)
    #     cv2.waitKey(100)
    images = images.reshape((-1, 8 * 8))[:n]
    return images

def make_spiral(n=100):
    theta = np.radians(np.linspace(90, 360 * 4, n))
    theta *= np.geomspace(1, 2.4, n)[::-1]
    r = theta ** 2
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.concatenate([x[..., None], y[..., None]], axis=1)


def make_2_spiral(n=100):
    n = n // 2
    theta = np.sqrt(np.random.rand(n))*2*pi

    r_a = 2*theta + pi
    data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
    x_a = data_a + np.random.randn(n,2)

    r_b = -2*theta - pi
    data_b = np.array([np.cos(theta)*r_b, np.sin(theta)*r_b]).T
    x_b = data_b + np.random.randn(n,2)

    return np.append(x_a, x_b, axis=0)
