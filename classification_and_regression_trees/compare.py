#!/usr/bin/env python
# -*- coding: utf-8 -*-

from regression_tree import *
from model_tree import linear_regression

def get_corrcoef(X, Y):
    # X Y 的协方差
    cov = np.mean(X*Y) - np.mean(X)*np.mean(Y)
    return cov/(np.var(X)*np.var(Y))**0.5

if '__main__' == __name__:
    # 加载数据
    data_train = load_data('bikeSpeedVsIq_train.txt')
    data_test = load_data('bikeSpeedVsIq_test.txt')

    dataset_test = np.matrix(data_test)
    m, n = dataset_test.shape
    testset = np.ones((m, n+1))
    testset[:, 1:] = dataset_test
    X, y = testset[:, :-1], testset[:, -1]

    # 获取标准线性回归模型
    w, X, y = linear_regression(data_train)
    y_lr = X*w
    y = np.array(y).T[0]
    y_lr = np.array(y_lr).T[0]
    corrcoef_lr = get_corrcoef(y, y_lr)
    print('linear regression correlation coefficient: {}'.format(corrcoef_lr))

    # 获取模型树回归模型
    tree = create_tree(data_train, fleaf, ferr, opt={'err_tolerance': 1,
                                                     'n_tolerance': 4})
    y_tree = [tree_predict(x, tree) for x in X[:, 1].tolist()]
    corrcoef_tree = get_corrcoef(np.array(y_tree), y)
    print('regression tree correlation coefficient: {}'.format(corrcoef_tree))

    plt.scatter(np.array(data_train)[:, 0], np.array(data_train)[:, 1])
    #plt.show()

