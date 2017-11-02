#!/usr/bin/env python
# -*- coding: utf-8 -*-

from regression_tree import *

def linear_regression(dataset):
    ''' 获取标准线性回归系数
    '''
    dataset = np.matrix(dataset)
    # 分割数据并添加常数列
    X_ori, y = dataset[:, :-1], dataset[:, -1]
    X_ori, y = np.matrix(X_ori), np.matrix(y)
    m, n = X_ori.shape
    X = np.matrix(np.ones((m, n+1)))
    X[:, 1:] = X_ori

    # 回归系数
    w = (X.T*X).I*X.T*y
    return w, X, y

def fleaf(dataset):
    ''' 计算给定数据集的线性回归系数
    '''
    w, _, _ = linear_regression(dataset)
    return w

def ferr(dataset):
    ''' 对给定数据集进行回归并计算误差
    '''
    w, X, y = linear_regression(dataset)
    y_prime = X*w
    return np.var(y_prime - y)

def tree_predict(data, tree):
    if type(tree) is not dict:
        w = tree
        y = np.matrix(data)*w
        return y[0, 0]

    feat_idx, feat_val = tree['feat_idx'], tree['feat_val']
    if data[feat_idx+1] < feat_val:
        return tree_predict(data, tree['left'])
    else:
        return tree_predict(data, tree['right'])

if '__main__' == __name__:
    dataset = load_data('exp2.txt')
    tree = create_tree(dataset, fleaf, ferr, opt={'err_tolerance': 0.1, 'n_tolerance': 4})

    dataset = np.array(dataset)
    # 绘制散点图
    plt.scatter(dataset[:, 0], dataset[:, 1])

    # 绘制回归曲线
    x = np.sort(dataset[:, 0])
    y = [tree_predict([1.0] + [i], tree) for i in x]
    plt.plot(x, y, c='r')
    plt.show()

