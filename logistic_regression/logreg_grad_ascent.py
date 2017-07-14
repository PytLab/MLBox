#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from math import exp

import numpy as np
import matplotlib.pyplot as plt

class LogisticRegressionClassifier(object):
    ''' 使用梯度上升算法Logistic回归分类器
    '''

    @staticmethod
    def sigmoid(x):
        ''' Sigmoid 阶跃函数
        '''
        return 1.0/(1 + np.exp(-x))

    def gradient_ascent(self, dataset, labels, max_iter=10000):
        ''' 使用梯度上升优化Logistic回归模型参数

        :param dataset: 数据特征矩阵
        :type dataset: MxN numpy matrix

        :param labels: 数据集对应的类型向量
        :type labels: Nx1 numpy matrix
        '''
        dataset = np.matrix(dataset)
        vlabels = np.matrix(labels).reshape(-1, 1)
        m, n = dataset.shape
        w = np.ones((n, 1))
        alpha = 0.001
        ws = []
        for i in range(max_iter):
            error = vlabels - self.sigmoid(dataset*w)
            w += alpha*dataset.T*error
            ws.append(w.reshape(1, -1).tolist()[0])

        return w, np.array(ws)

def load_data(filename):
    dataset, labels = [], []
    with open(filename, 'r') as f:
        for line in f:
            splited_line = [float(i) for i in line.strip().split('\t')]
            data, label = [1.0] + splited_line[: -1], splited_line[-1]
            dataset.append(data)
            labels.append(label)
    dataset = np.array(dataset)
    labels = np.array(labels)

    return dataset, labels

def snapshot(w, dataset, labels, pic_name):
    ''' 绘制类型分割线图
    '''
    if not os.path.exists('./snapshots'):
        os.mkdir('./snapshots')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    pts = {}
    for data, label in zip(dataset.tolist(), labels.tolist()):
        pts.setdefault(label, [data]).append(data)

    for label, data in pts.items():
        data = np.array(data)
        plt.scatter(data[:, 1], data[:, 2], label=label, alpha=0.5)

    # 分割线绘制
    def get_y(x, w):
        w0, w1, w2 = w
        return (-w0 - w1*x)/w2

    x = [-4.0, 3.0]
    y = [get_y(i, w) for i in x]

    plt.plot(x, y, linewidth=2, color='#FB4A42')

    pic_name = './snapshots/{}'.format(pic_name)
    fig.savefig(pic_name)
    plt.close(fig)

if '__main__' == __name__:
    clf = LogisticRegressionClassifier()
    dataset, labels = load_data('testSet.txt')
    w, ws = clf.gradient_ascent(dataset, labels, max_iter=50000)
    m, n = ws.shape

    # 绘制分割线
    for i in range(300):
        if i % (30) == 0:
            print('{}.png saved'.format(i))
            snapshot(ws[i].tolist(), dataset, labels, '{}.png'.format(i))

    fig = plt.figure()
    for i in range(n):
        label = 'w{}'.format(i)
        ax = fig.add_subplot(n, 1, i+1)
        ax.plot(ws[:, i], label=label)
        ax.legend()

    fig.savefig('w_traj.png')

