#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random

import numpy as np
import matplotlib.pyplot as plt


class SVMUtil(object):
    '''
    Struct to save all important values in SVM.
    '''
    def __init__(self, dataset, labels, C, tolerance=0.001):
        self.dataset, self.labels, self.C = dataset, labels, C

        self.m, self.n = np.array(dataset).shape
        self.alphas = np.zeros(self.m)
        self.b = 0
        self.tolerance = tolerance
        # Cached errors ,f(x_i) - y_i
        self.errors = [self.get_error(i) for i in range(self.m)]

    def f(self, x):
        '''SVM分类器函数 y = w^Tx + b
        '''
        # Kernel function vector.
        x = np.matrix(x).T
        data = np.matrix(self.dataset)
        ks = data*x

        # Predictive value.
        wx = np.matrix(self.alphas*self.labels)*ks
        fx = wx + self.b

        return fx[0, 0]

    def get_error(self, i):
        ''' 获取第i个数据对应的误差.
        '''
        x, y = self.dataset[i], self.labels[i]
        fx = self.f(x)
        return fx - y

    def update_errors(self):
        ''' 更新所有的误差值.
        '''
        self.errors = [self.get_error(i) for i in range(self.m)]

    def meet_kkt(self, i):
        alpha = self.alphas[i]
        x = self.dataset[i]

        if alpha == 0:
            return self.f(x) >= 1
        elif alpha == self.C:
            return self.f(x) <= 1
        else:
            return self.f(x) == 1

def load_data(filename):
    dataset, labels = [], []
    with open(filename, 'r') as f:
        for line in f:
            x, y, label = [float(i) for i in line.strip().split()]
            dataset.append([x, y])
            labels.append(label)
    return dataset, labels

def clip(alpha, L, H):
    ''' 修建alpha的值到L和H之间.
    '''
    if alpha < L:
        return L
    elif alpha > H:
        return H
    else:
        return alpha

def select_j_rand(i, m):
    ''' 在m中随机选择除了i之外剩余的数
    '''
    l = list(range(m))
    seq = l[: i] + l[i+1:]
    return random.choice(seq)

def select_j(i, svm_util):
    ''' 通过最大化步长的方式来获取第二个alpha值的索引.
    '''
    errors = svm_util.errors
    valid_indices = [i for i, a in enumerate(svm_util.alphas) if 0 < a < svm_util.C]

    if len(valid_indices) > 1:
        j = -1
        max_delta = 0
        for k in valid_indices:
            if k == i:
                continue
            delta = abs(errors[i] - errors[j])
            if delta > max_delta:
                j = k
                max_delta = delta
    else:
        j = select_j_rand(i, svm_util.m)
    return j

def get_w(alphas, dataset, labels):
    ''' 通过已知数据点和拉格朗日乘子获得分割超平面参数w
    '''
    alphas, dataset, labels = np.array(alphas), np.array(dataset), np.array(labels)
    yx = labels.reshape(1, -1).T*np.array([1, 1])*dataset
    w = np.dot(yx.T, alphas)

    return w.tolist()

def take_step(i, j, svm_util):
    ''' 对选定的一对alpha对进行优化.
    '''
    svm_util.update_errors()
    alphas, dataset, labels = svm_util.alphas, svm_util.dataset, svm_util.labels
    C, b = svm_util.C, svm_util.b

    a_i, x_i, y_i, E_i = alphas[i], dataset[i], labels[i], svm_util.errors[i]
    a_j, x_j, y_j, E_j = alphas[j], dataset[j], labels[j], svm_util.errors[j]

    K_ii, K_jj, K_ij = np.dot(x_i, x_i), np.dot(x_j, x_j), np.dot(x_i, x_j)
    eta = K_ii + K_jj - 2*K_ij
    if eta <= 0:
        print('WARNING  eta <= 0')
        return 0

    a_i_old, a_j_old = a_i, a_j
    a_j_new = a_j_old + y_j*(E_i - E_j)/eta
            
    # 对alpha进行修剪
    if y_i != y_j:
        L = max(0, a_j_old - a_i_old)
        H = min(C, C + a_j_old - a_i_old)
    else:
        L = max(0, a_i_old + a_j_old - C)
        H = min(C, a_j_old + a_i_old)

    a_j_new = clip(a_j_new, L, H)
    a_i_new = a_i_old + y_i*y_j*(a_j_old - a_j_new)

    if abs(a_j_new - a_j_old) < 0.00001:
        #print('WARNING   alpha_j not moving enough')
        return 0

    alphas[i], alphas[j] = a_i_new, a_j_new
    svm_util.update_errors()

    # 更新阈值b
    b_i = -E_i - y_i*K_ii*(a_i_new - a_i_old) - y_j*K_ij*(a_j_new - a_j_old) + b
    b_j = -E_j - y_i*K_ij*(a_i_new - a_i_old) - y_j*K_jj*(a_j_new - a_j_old) + b

    if 0 < a_i_new < C:
        b = b_i
    elif 0 < a_j_new < C:
        b = b_j
    else:
        b = (b_i + b_j)/2

    svm_util.b = b
    print(b)

    return 1

def examine_example(i, svm_util):
    ''' 给定第一个alpha， 检测对应alpha是否符合KKT条件并选取第二个alpha进行迭代.
    '''
    E_i, y_i, alpha = svm_util.errors[i], svm_util.labels[i], svm_util.alphas[i]
    r = E_i*y_i
    C, tolerance = svm_util.C, svm_util.tolerance

    # 是否违反KKT条件
    if (r < -tolerance and alpha < C) or (r > tolerance and alpha > 0):
        j = select_j(i, svm_util)
        #j = select_j_rand(i, svm_util.m)
        return take_step(i, j, svm_util)
    else:
        return 0

def platt_smo(dataset, labels, C, max_iter):
    ''' Platt SMO算法实现，使用启发式方法对alpha对进行选择.

    :param dataset: 所有特征数据向量
    :param labels: 所有的数据标签
    :param C: 软间隔常数, 0 <= alpha_i <= C
    :param max_iter: 外层循环最大迭代次数
    '''
    # 初始化SVM工具对象
    svm_util = SVMUtil(dataset, labels, C)
    it = 0

    # 遍历所有alpha的标记
    entire = True

    pair_changed = 0
    while (it < max_iter): #and (pair_changed > 0 or entire):
        pair_changed = 0
        if entire:
            for i in range(svm_util.m):
                pair_changed += examine_example(i, svm_util)
                print('Full set - iter: {}, pair changed: {}'.format(i, pair_changed))
        else:
            alphas = svm_util.alphas
            non_bound_indices = [i for i in range(svm_util.m)
                                 if alphas[i] > 0 and alphas[i] < C]
            for i in non_bound_indices:
                pair_changed += examine_example(i, svm_util)
                print('Non-bound - iter:{}, pair changed: {}'.format(i, pair_changed))
        it += 1

        if entire:
            entire = False
        elif pair_changed == 0:
            entire = True

        print('iteration number: {}'.format(it))

    return svm_util.alphas, svm_util.b

if '__main__' == __name__:
    # 加载训练数据
    dataset, labels = load_data('testSet.txt')
    # 使用简化版SMO算法优化SVM
    alphas, b = platt_smo(dataset, labels, 0.8, 40)

    # 分类数据点
    classified_pts = {'+1': [], '-1': []}
    for point, label in zip(dataset, labels):
        if label == 1.0:
            classified_pts['+1'].append(point)
        else:
            classified_pts['-1'].append(point)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # 绘制数据点
    for label, pts in classified_pts.items():
        pts = np.array(pts)
        ax.scatter(pts[:, 0], pts[:, 1], label=label)

    # 绘制分割线
    w = get_w(alphas, dataset, labels)
    x1, _ = max(dataset, key=lambda x: x[0])
    x2, _ = min(dataset, key=lambda x: x[0])
    a1, a2 = w
    y1, y2 = (-b - a1*x1)/a2, (-b - a1*x2)/a2
    ax.plot([x1, x2], [y1, y2])

    # 绘制支持向量
    for i, alpha in enumerate(alphas):
        if abs(alpha) > 1e-3:
            x, y = dataset[i]
            ax.scatter([x], [y], s=150, c='none', alpha=0.7,
                       linewidth=1.5, edgecolor='#AB3319')

    plt.show()

