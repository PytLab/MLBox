#!/usr/bin/env python
# -*- coding: utf-8 -*-

from trees import DecisionTreeClassifier

lense_labels = ['age', 'prescript', 'astigmatic', 'tearRate']
X = []
Y = []

with open('lenses.txt', 'r') as f:
    for line in f:
        comps = line.strip().split('\t')
        X.append(comps[: -1])
        Y.append(comps[-1])

clf = DecisionTreeClassifier()
clf.create_tree(X, Y, lense_labels)

