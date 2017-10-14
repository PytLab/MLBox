#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random

import numpy as np
import matplotlib.pyplot as plt

from gaft import GAEngine
from gaft.components import GAIndividual, GAPopulation
from gaft.operators import RouletteWheelSelection, UniformCrossover, FlipBitBigMutation

from gaft.analysis.fitness_store import FitnessStore
from gaft.analysis.console_output import ConsoleOutput


def load_data(filename):
    dataset, labels = [], []
    with open(filename, 'r') as f:
        for line in f:
            x, y, label = [float(i) for i in line.strip().split()]
            dataset.append([x, y])
            labels.append(label)
    return dataset, labels

def get_w(alphas, dataset, labels):
    ''' 通过已知数据点和拉格朗日乘子获得分割超平面参数w
    '''
    alphas, dataset, labels = np.array(alphas), np.array(dataset), np.array(labels)
    yx = labels.reshape(1, -1).T*np.array([1, 1])*dataset
    w = np.dot(yx.T, alphas)

    return w.tolist()

# Population definition.
indv_template = GAIndividual(ranges=[(-2, 2), (-2, 2), (-5, 5)],
                             encoding='binary',
                             eps=[0.001, 0.001, 0.005])
population = GAPopulation(indv_template=indv_template, size=200).init()

# Genetic operators.
selection = RouletteWheelSelection()
crossover = UniformCrossover(pc=0.8, pe=0.5)
mutation = FlipBitBigMutation(pm=0.1, pbm=0.55, alpha=0.6)

engine = GAEngine(population=population, selection=selection,
                  crossover=crossover, mutation=mutation,
                  analysis=[ConsoleOutput, FitnessStore])

# 加载数据
dataset, labels = load_data('testSet.txt')

@engine.fitness_register
def fitness(indv):
    w, b = indv.variants[: -1], indv.variants[-1]
    min_dis = float('inf')
    for x, y in zip(dataset, labels):
        dis = y*(np.dot(w, x) + b)
        if dis < min_dis:
            min_dis = dis
    return float(min_dis)

if '__main__' == __name__:
    engine.run(500)

    variants = engine.population.best_indv(engine.fitness).variants
    w = variants[: -1]
    b = variants[-1]

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
    x1, _ = max(dataset, key=lambda x: x[0])
    x2, _ = min(dataset, key=lambda x: x[0])
    a1, a2 = w
    y1, y2 = (-b - a1*x1)/a2, (-b - a1*x2)/a2
    ax.plot([x1, x2], [y1, y2])

    plt.show()

