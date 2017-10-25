#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import exp

import numpy as np
import matplotlib.pyplot as plt

from standard_linear_regression import load_data, get_corrcoef

def ridge_regression(X, y, lambd=0.2):
    ''' 获取岭回归系数
    '''
    XTX = X.T*X
    m, _ = XTX.shape
    I = np.matrix(np.eye(m))
    w = (XTX + lambd*I).I*X.T*y
    return w

def standarize(X):
    ''' 中心化 & 标准化数据 (零均值, 单位标准差)
    '''
    std_deviation = np.std(X, 0)
    mean = np.mean(X, 0)
    return (X - mean)/std_deviation

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
    X, y = load_data('ex0.txt')
    ws = ridge_traj(X, y, ntest)

    # 绘制岭轨迹
    fig = plt.figure()
    ax = fig.add_subplot(111)

    lambdas = [i-10 for i in range(ntest)]
    ax.plot(lambdas, ws)

    plt.show()

