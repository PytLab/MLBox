#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import exp

import numpy as np
import matplotlib.pyplot as plt

from standard_linear_regression import load_data, get_corrcoef

def lwlr(x, X, Y, k):
    ''' 局部加权线性回归，给定一个点，获取相应权重矩阵并返回回归系数
    '''
    m = X.shape[0]

    # 创建针对x的权重矩阵
    W = np.matrix(np.zeros((m, m)))
    for i in range(m):
        xi = np.array(X[i][0])
        x = np.array(x)
        W[i, i] = exp((np.linalg.norm(x - xi))/(-2*k**2))

    # 获取此点相应的回归系数

    xWx = X.T*W*X
    if np.linalg.det(xWx) == 0:
        print('xWx is a singular matrix')
        return
    w = xWx.I*X.T*W*Y

    return w

if '__main__' == __name__:
    k = 0.03

    X, Y = load_data('ex0.txt')
    
    y_prime = []
    for x in X.tolist():
        w = lwlr(x, X, Y, k).reshape(1, -1).tolist()[0]
        y_prime.append(np.dot(x, w))

    corrcoef = get_corrcoef(np.array(Y.reshape(1, -1)), np.array(y_prime))
    print('Correlation coefficient: {}'.format(corrcoef))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # 绘制数据点
    x = X[:, 1].reshape(1, -1).tolist()[0]
    y = Y.reshape(1, -1).tolist()[0]
    ax.scatter(x, y)

    # 绘制拟合直线
    x, y = list(zip(*sorted(zip(x, y_prime), key=lambda x: x[0])))
    ax.plot(x, y, c='r')

    plt.show()

