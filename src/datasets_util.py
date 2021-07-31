import os
import math
import cv2
import numpy as np
import pickle
from sklearn.feature_extraction import image
from numpy import pi
from mnist import MNIST


def load_mnist(path, digit, n=2000):
    mndata = MNIST(path)
    images, labels = mndata.load_training()
    images = np.array(images)
    labels = np.array(labels)

    # Create balanced subset
    v = images[labels == digit]
    X = v[np.random.choice(v.shape[0], size=n, replace=False)]
    assert X.shape[0] == n
    return X.T


def load_vidtimit(path, subject=0, skip_pickle=False):
    pickle_path = os.path.join(path, 'vidtimit_%s.pickle' % subject)
    
    if not skip_pickle and os.path.isfile(pickle_path):
        with open(pickle_path, 'rb') as f:
            images = pickle.load(f)
    else:
        print('Vidtimit pickle not found, creating it')
        face_cascade = cv2.CascadeClassifier('../data/haarcascade_frontalface_default.xml')
        profile_cascade = cv2.CascadeClassifier('../data/haarcascade_profileface.xml')

        images = []
        folders = sorted([p for p in os.listdir(path) if os.path.isdir(os.path.join(path, p))])
        for sub in folders[subject:subject + 1]:
            p = os.path.join(os.path.join(path, sub), 'video')
            if not os.path.isdir(p):
                continue
            folders = ['head', 'head2', 'head3']
            for fold in folders:
                pp = os.path.join(p, fold)
                for f in os.listdir(pp):
                    img = cv2.imread(os.path.join(pp, f))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(img, 1.3, 5)

                    if len(faces) == 0:
                        faces = profile_cascade.detectMultiScale(img, 1.3, 5)
                        if len(faces) == 0:
                            continue
                        continue
                    x, y, w, h = faces[0]
                    if w < 100 or h < 100:
                        continue

                    # img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                    # cv2.imshow('', img)

                    img = img[y: y + h, x:x + w]
                    img = cv2.resize(img, (26, 26), interpolation=cv2.INTER_CUBIC)

                    # cv2.imshow('crop', img)
                    # cv2.waitKey(1)
                    images.append(img)

        with open(pickle_path, 'wb') as f:
            pickle.dump(images, f, protocol=pickle.HIGHEST_PROTOCOL)

    # for img in images:
    #     cv2.imshow('', cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    #     cv2.waitKey(100)

    images = np.array(images).reshape((-1, 26 * 26))
    return images


def load_bsds(path, n=10000):
    images = [cv2.imread(os.path.join(path, f)) for f in os.listdir(path)]
    # images = images[:1]
    # images = [cv2.resize(im, (im.shape[1] // 16, im.shape[0] // 16)) for im in images]
    images = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in images]
    images = np.array([image.extract_patches_2d(im, (8, 8), max_patches=n // len(images) + 1) for im in images])
    images = np.reshape(images, (-1, 8, 8))
    # for img in images:
    #     cv2.imshow('', cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    #     cv2.waitKey(100)
    images = images.reshape((-1, 8 * 8))[:n]
    assert len(images) == n
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
