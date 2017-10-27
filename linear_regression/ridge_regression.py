#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import exp

import numpy as np
import matplotlib.pyplot as plt

from standard_linear_regression import load_data, get_corrcoef, standarize

def ridge_regression(X, y, lambd=0.2):
    ''' 获取岭回归系数
    '''
    XTX = X.T*X
    m, _ = XTX.shape
    I = np.matrix(np.eye(m))
    w = (XTX + lambd*I).I*X.T*y
    return w

def ridge_traj(X, y, ntest=30):
    ''' 获取岭轨迹矩阵
    '''
    _, n = X.shape
    ws = np.zeros((ntest, n))
    for i in range(ntest):
        w = ridge_regression(X, y, lambd=exp(i-10))
        ws[i, :] = w.T
    return ws

if '__main__' == __name__:
    ntest = 30
    # 加载数据
    X, y = load_data('abalone.txt')

    # 中心化 & 标准化
    X, y = standarize(X), standarize(y)

    # 测试数据和训练数据
    w_test, errors = [], []
    for i in range(ntest):
        lambd = exp(i - 10)
        # 训练数据
        X_train, y_train = X[: 180, :], y[: 180, :]
        # 测试数据
        X_test, y_test = X[180: -1, :], y[180: -1, :]

        # 岭回归系数
        w = ridge_regression(X_train, y_train, lambd)
        error = np.std(X_test*w - y_test)
        w_test.append(w)
        errors.append(error)

    # 选择误差最小的回归系数
    w_best, e_best = min(zip(w_test, errors), key=lambda x: x[1])
    print('Best w: {}, best error: {}'.format(w_best, e_best))

    y_prime = X*w_best
    # 计算相关系数
    corrcoef = get_corrcoef(np.array(y.reshape(1, -1)),
                            np.array(y_prime.reshape(1, -1)))
    print('Correlation coefficient: {}'.format(corrcoef))

    # 绘制岭轨迹
    ws = ridge_traj(X, y, ntest)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    lambdas = [i-10 for i in range(ntest)]
    ax.plot(lambdas, ws)

    plt.show()

