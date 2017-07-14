#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random

import numpy as np
import matplotlib.pyplot as plt

from logreg_grad_ascent import LogisticRegressionClassifier as BaseClassifer
from logreg_grad_ascent import load_data, snapshot

class LogisticRegressionClassifier(BaseClassifer):

    def random_gradient_ascent(self, dataset, labels, max_iter=150):
        ''' 使用随机梯度上升算法优化Logistic回归模型参数
        '''
        dataset = np.matrix(dataset)
        m, n = dataset.shape
        w = np.matrix(np.ones((n, 1)))
        ws = []

        for i in range(max_iter):
            data_indices = list(range(m))
            random.shuffle(data_indices)
            for j, idx in enumerate(data_indices):
                data, label = dataset[idx], labels[idx]
                error = label - self.sigmoid((data*w).tolist()[0][0])
                alpha = 4/(1 + j + i) + 0.01
                w += alpha*data.T*error
                ws.append(w.T.tolist()[0])

        return w, np.array(ws)

if '__main__' == __name__:
    clf = LogisticRegressionClassifier()
    dataset, labels = load_data('testSet.txt')
    w, ws = clf.random_gradient_ascent(dataset, labels, max_iter=500)
    m, n = ws.shape

    # 绘制分割线
    for i, w in enumerate(ws):
        if i % (m//10) == 0:
            print('{}.png saved'.format(i))
            snapshot(w.tolist(), dataset, labels, '{}.png'.format(i))

    fig = plt.figure()
    for i in range(n):
        label = 'w{}'.format(i)
        ax = fig.add_subplot(n, 1, i+1)
        ax.plot(ws[:, i], label=label)
        ax.legend()

    fig.savefig('rand_grad_ascent_params.png')
