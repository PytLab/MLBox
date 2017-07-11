#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' 通过垃圾短信数据训练朴素贝叶斯模型，并进行留存交叉验证
'''

import re
import random

import numpy as np
import matplotlib.pyplot as plt

from bayes import NaiveBayesClassifier

ENCODING = 'ISO-8859-1'
TRAIN_PERCENTAGE = 0.9

def get_doc_vector(words, vocabulary):
    ''' 根据词汇表将文档中的词条转换成文档向量

    :param words: 文档中的词条列表
    :type words: list of str

    :param vocabulary: 总的词汇列表
    :type vocabulary: list of str

    :return doc_vect: 用于贝叶斯分析的文档向量
    :type doc_vect: list of int
    '''
    doc_vect = [0]*len(vocabulary)

    for word in words:
        if word in vocabulary:
            idx = vocabulary.index(word)
            doc_vect[idx] = 1

    return doc_vect

def parse_line(line):
    ''' 解析数据集中的每一行返回词条向量和短信类型.
    '''
    cls = line.split(',')[-1].strip()
    content = ','.join(line.split(',')[: -1])
    word_vect = [word.lower() for word in re.split(r'\W+', content) if word]
    return word_vect, cls

def parse_file(filename):
    ''' 解析文件中的数据
    '''
    vocabulary, word_vects, classes = [], [], []
    with open(filename, 'r', encoding=ENCODING) as f:
        for line in f:
            if line:
                word_vect, cls = parse_line(line)
                vocabulary.extend(word_vect)
                word_vects.append(word_vect)
                classes.append(cls)
    vocabulary = list(set(vocabulary))

    return vocabulary, word_vects, classes

if '__main__' == __name__:
    clf = NaiveBayesClassifier()
    vocabulary, word_vects, classes = parse_file('english_big.txt')

    # 训练数据 & 测试数据
    ntest = int(len(classes)*(1-TRAIN_PERCENTAGE))

    test_word_vects = []
    test_classes = []
    for i in range(ntest):
        idx = random.randint(0, len(word_vects)-1)
        test_word_vects.append(word_vects.pop(idx))
        test_classes.append(classes.pop(idx))

    train_word_vects = word_vects
    train_classes = classes

    train_dataset = [get_doc_vector(words, vocabulary) for words in train_word_vects]

    # 训练贝叶斯模型
    cond_probs, cls_probs = clf.train(train_dataset, train_classes)

    # 测试模型
    error = 0
    for test_word_vect, test_cls in zip(test_word_vects, test_classes):
        test_data = get_doc_vector(test_word_vect, vocabulary)
        pred_cls = clf.classify(test_data, cond_probs, cls_probs)
        if test_cls != pred_cls:
            print('Predict: {} -- Actual: {}'.format(pred_cls, test_cls))
            error += 1

    print('Error Rate: {}'.format(error/len(test_classes)))

    # 绘制不同类型的概率分布曲线
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for cls, probs in cond_probs.items():
        ax.scatter(np.arange(0, len(probs)),
                   probs*cls_probs[cls],
                   label=cls,
                   alpha=0.3)
        ax.legend()

    plt.show()

