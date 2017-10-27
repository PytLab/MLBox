#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from standard_linear_regression import load_data, get_corrcoef
from standard_linear_regression import standarize

def stagewise_regression(X, y, eps=0.01, niter=100):
    ''' 通过向前逐步回归获取回归系数
    '''
    m, n = X.shape
    w = np.matrix(np.zeros((n, 1)))
    min_error = float('inf')
    all_ws = np.matrix(np.zeros((niter, n)))

    # 计算残差平方和
    rss = lambda X, y, w: (y - X*w).T*(y - X*w)

    for i in range(niter):
        print('{}: w = {}'.format(i, w.T[0, :]))
        for j in range(n):
            for sign in [-1, 1]:
                w_test = w.copy()
                w_test[j, 0] += eps*sign
                test_error = rss(X, y, w_test)
                if test_error < min_error:
                    min_error = test_error
                    w = w_test
        all_ws[i, :] = w.T

    return all_ws

if '__main__' == __name__:
    X, y = load_data('abalone.txt')
    X, y = standarize(X), standarize(y)

    epsilon = 0.005
    niter = 1000
    all_ws = stagewise_regression(X, y, eps=epsilon, niter=niter)

    w = all_ws[-1, :]
    y_prime = X*w.T

    # 计算相关系数
    corrcoef = get_corrcoef(np.array(y.reshape(1, -1)),
                            np.array(y_prime.reshape(1, -1)))
    print('Correlation coefficient: {}'.format(corrcoef))

    # 绘制逐步线性回归回归系数变化轨迹

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(list(range(niter)), all_ws)
    plt.show()

