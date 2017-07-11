#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict

import numpy as np

class NaiveBayesClassifier(object):
    ''' 朴素贝叶斯分类器
    '''

    def train(self, dataset, classes):
        ''' 训练朴素贝叶斯模型

        :param dataset: 所有的文档数据向量
        :type dataset: MxN matrix containing all doc vectors.

        :param classes: 所有文档的类型
        :type classes: 1xN list

        :return cond_probs: 训练得到的条件概率矩阵
        :type cond_probs: MxK matrix

        :return cls_probs: 各种类型的概率
        :type cls_probs: 1xK list
        '''
        # 按照不同类型记性分类
        sub_datasets = defaultdict(lambda: [])
        cls_cnt = defaultdict(lambda: 0)

        for doc_vect, cls in zip(dataset, classes):
            sub_datasets[cls].append(doc_vect)
            cls_cnt[cls] += 1

        # 计算类型概率
        cls_probs = {k: v/len(classes) for k, v in cls_cnt.items()}

        # 计算不同类型的条件概率
        cond_probs = {}
        dataset = np.array(dataset)
        for cls, sub_dataset in sub_datasets.items():
            sub_dataset = np.array(sub_dataset)
            # Improve the classifier.
            cond_prob_vect = np.log((np.sum(sub_dataset, axis=0) + 1)/(np.sum(dataset) + 2))
            cond_probs[cls] = cond_prob_vect

        return cond_probs, cls_probs

    def classify(self, doc_vect, cond_probs, cls_probs):
        ''' 使用朴素贝叶斯将doc_vect进行分类.
        '''
        pred_probs = {}
        for cls, cls_prob in cls_probs.items():
            cond_prob_vect = cond_probs[cls]
            pred_probs[cls] = np.sum(cond_prob_vect*doc_vect) + np.log(cls_prob)
        return max(pred_probs, key=pred_probs.get)

