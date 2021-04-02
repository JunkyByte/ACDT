import os
import numpy as np
import scipy.io
from scipy.linalg import svd, orth, null_space
from multiprocessing import Pool
from numba import njit, extending, jit
PROCESS = os.cpu_count()


def karcher_mean(P, m, eps, maxiters):
    n = P[0].shape[0]
    k = P[0].shape[1]
    p_bar = P[0]

    nw = np.inf
    w = np.zeros((n, k))

    lowest_nw = np.inf
    best_p_bar = p_bar
    iters = 0
    while nw > eps:
        U, S, Vh = svd(w, full_matrices=False, overwrite_a=True, check_finite=False)
        V = Vh.T
        p_bar = np.dot(np.dot(p_bar, V), np.diag(np.cos(S))) + np.dot(U, np.diag(np.sin(S)))
        w = np.zeros((n, k))

        # Candy multiprocessing for parallel logq_map
        for x in pool.starmap(logq_map, [(p_bar, point) for point in P]):
            w = w + x

        w = w / m

        nw = np.linalg.norm(w, ord='fro')

        if iters > maxiters:
            print('Max iters reached')
            break

        lowest_nw = nw
        best_p_bar = p_bar
        iters += 1
    print('Error on out: %s' % nw)
    return orth(best_p_bar)


def logq_map(p, q):
    """
    Compute the Log_p(q) map for Grassman Manifold where
    p = point of tancency. p,q in G(k, n) with p'p = q'q = I
    """
    A = np.dot(p.T, q)
    n, k = p.shape
    B = np.dot(null_space(p.T).T, q)
    V, W, Z, C, S = cs_decomp(A, B)
    if n > 2 * k:
        S = S[:k, :]
        W = W[:, :k]
    elif n < 2 * k:
        S = np.vstack([S, np.zeros((2 * k - n, k))])
        W = np.hstack([W, np.zeros((n - k, 2 * k - n))])
    C = np.diag(1 / np.diag(C))
    U = np.dot(null_space(p.T), W)
    T = np.arctan(np.dot(S, C))
    X = np.dot(np.dot(U, T), V.T)
    return X


def cs_decomp(Q1, Q2):
    m, p = Q1.shape
    n, pb = Q2.shape

    if m < n:
        V, U, Z, S, C = cs_decomp(Q2, Q1)
        j = range(p - 1, -1, -1)
        C = C[:, j]
        S = S[:, j]
        Z = Z[:, j]
        m = min(m, p)
        i = range(m - 1, -1, -1)
        C[:m, :] = C[i, :]
        U[:, :m] = U[:, i]
        n = min(n, p)
        i = range(n - 1, -1, -1)
        S[:n, :] = S[i, :]
        V[:, :n] = V[:, i]
        return U, V, Z, C, S

    U, C, Zh = svd(Q1, overwrite_a=True, check_finite=False)
    C = np.diag(C)
    Z = Zh.T

    q = min(m, p)
    i = range(q)
    j = range(q - 1, -1, -1)
    C[i, i] = C[j, j]
    U[:, i] = U[:, j]
    Z[:, i] = Z[:, j]
    S = np.dot(Q2, Z)

    if q == 1:
        k = 0
    elif m < p:
        k = n
    else:
        k = np.where(np.diag(C) <= 1 / np.sqrt(2))[0]
        if k.size == 0:
            k = 0  # TODO
        else:
            k = np.max(k) + 1
    if k != 0:
        V, R = np.linalg.qr(S[:, :k], mode='complete')
    else:
        V = np.eye(S.shape[0])

    S = np.dot(V.T, S)
    r = min(k, m)
    S[:, :r] = diagf(S[:, :r])
    if m == 1 and p > 1:
        S[0, 0] = 0
    if k < min(n, p):
        r = min(n, p)
        i = range(k, n)
        j = range(k, r)
        if S[i, j].shape == (1,):
            UT = 1
            ST = S[0, 0]
            VT = 1
        else:
            UT, ST, VTh = svd(S[np.ix_(i, j)], check_finite=False)
            ST = np.diag(ST)
            VT = VTh.T
        if k > 0:
            S[:k, j] = 0

        S[np.ix_(i, j)] = ST
        C[:, j] = np.dot(C[:, j], VT)
        V[:, i] = np.dot(V[:, i], UT)
        Z[:, j] = np.dot(Z[:, j], VT)
        i = range(k, q)

        Q, R = np.linalg.qr(C[np.ix_(i, j)], mode='complete')
        C[np.ix_(i, j)] = diagf(R)
        U[:, i] = np.dot(U[:, i], Q)

    if m < p:
        a = np.count_nonzero(np.abs(diagk(C, 0)) > 10 * m * np.finfo(C.dtype).resolution)
        b = np.count_nonzero(np.abs(diagk(S, 0)) > 10 * n * np.finfo(C.dtype).resolution)
        q = min(a, b)
        i = range(q + 1, n)
        j = range(m + 1, p)

        Q, R = np.linalg.qr(S[i, j], mode='complete')
        S[:, q + 1:p] = 0
        S[i, j] = diagf(R)
        V[:, i] = np.dot(V[:, i], Q)
        if n > 1:
            i = []
            i.extend(list(range(q + 1, q + p - m)))
            i.extend(list(range(1, q)))
            i.extend(list(range(q + p - m + 1, n)))
        else:
            i = 0
        j = []
        j.extend(list(range(m + 1, p)))
        j.extend(list(range(1, m)))
        C = C[:, j]
        S = S[i, j]
        Z = Z[:, j]
        V = V[:, i]
    if n < p:
        S[:, n:p] = 0

    U, C = diagp(U, C, max(0, p - m))
    C = np.real(C)
    V, S = diagp(V, S, 0)
    S = np.real(S)
    #print(U)
    #print(V)
    #print(Z)
    #print(C)
    #print(S)
    #print('----')
    return U, V, Z, C, S


@njit(cache=True)
def diagp(Y, X, k):
    D = diagk(X, k).astype(np.float64)
    j = np.where((np.real(D) < 0) | (np.imag(D) != 0))[0]
    #print(j.shape)
    if j.size != 0:
        D = np.diag(np.conjugate(D[j]) / np.abs(D[j]))
        Y[:, j] = np.dot(Y[:, j], D.T)
        #print(D.shape, X.shape)
        X[j, :] = np.dot(D, X[j, :])
    return Y, X


@njit(cache=True)
def diagf(X):
    return np.triu(np.tril(X))


@njit(cache=True)
def diagk(X, k):
    if min(np.shape(X)) > 1:
        return np.diag(X, k)
    elif 0 <= k and 1 + k <= X.shape[1]:
        return X[:, k]  # TODO
    elif k < 0 and 1 - k <= X.shape[0]:
        return X[:, -k]  # TODO
    return np.empty((0,), dtype=np.float64)


pool = Pool(processes=PROCESS)


if __name__ == '__main__':
    # n = 784  # R^n
    n = 10  # R^n
    k = 4  # d-dim
    m = 2 # num elements

    np.random.seed()

    x = []
    for i in range(m):
        x.append(np.random.randn(n, k))
    scipy.io.savemat('/Users/adryw//Documents/MATLAB/inp_out.mat', dict(x=x))

    # x = np.array([[2.7477, -0.1649, 2.7477, -0.1649],
    #               [0.7002, 0.0675, 0.7002, 0.0675],
    #               [0.2390, 0.1333, 0.2390, 0.1333]])

    import time
    t = time.time()

    print('INPUT')
    print(x)
    y = karcher_mean(x, m, 1e-16, 2000)
    print('INPUT')
    print(x)
    print('MEAN')
    print(y)

    y_hat = karcher_mean(x, m, 1e-4, 2000)
    y = karcher_mean(x, m, 1e-16, 2000)
    print(linalg.subspace_angles(y_hat, y))

    print(time.time() - t)
