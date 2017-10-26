#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
from math import exp

import numpy as np
import matplotlib.pyplot as plt

from standard_linear_regression import load_data, get_corrcoef
from ridge_regression import standarize


def lasso_regression(X, y, lambd=0.2, threshold=0.1):
    ''' 通过坐标下降(coordinate descent)法获取LASSO回归系数
    '''
    # 计算残差平方和
    rss = lambda X, y, w: (y - X*w).T*(y - X*w)

    # 初始化回归系数w.
    m, n = X.shape
    w = np.matrix(np.zeros((n, 1)))
    r = rss(X, y, w)

    # 使用坐标下降法优化回归系数w
    niter = itertools.count(1)

    for it in niter:
        for k in range(n):
            # 计算常量值z_k和p_k
            z_k = (X[:, k].T*X[:, k])[0, 0]
            p_k = 0
            for i in range(m):
                p_k += X[i, k]*(y[i, 0] - sum([X[i, j]*w[j, 0] for j in range(n) if j != k]))

            if p_k < -lambd/2:
                w_k = (p_k + lambd/2)/z_k
            elif p_k > lambd/2:
                w_k = (p_k - lambd/2)/z_k
            else:
                w_k = 0

            w[k, 0] = w_k

        r_prime = rss(X, y, w)
        delta = abs(r_prime - r)[0, 0]
        r = r_prime
        print('Iteration: {}, delta = {}'.format(it, delta))

        if delta < threshold:
            break

    return w

def lasso_traj(X, y, ntest=30):
    ''' 获取回归系数轨迹矩阵
    '''
    _, n = X.shape
    ws = np.zeros((ntest, n))
    for i in range(ntest):
        w = lasso_regression(X, y, lambd=exp(i-10))
        ws[i, :] = w.T
    return ws

if '__main__' == __name__:
    X, y = load_data('abalone.txt')
    X, y = standarize(X), standarize(y)
    w = lasso_regression(X, y, lambd=10)

#    ntest = 30
#
#    # 绘制岭轨迹
#    ws = lasso_traj(X, y, ntest)
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#
#    lambdas = [i-10 for i in range(ntest)]
#    ax.plot(lambdas, ws)
#
#    plt.show()

