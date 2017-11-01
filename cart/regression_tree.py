#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' 回归树实现
'''

import uuid
from functools import namedtuple

import numpy as np
import matplotlib.pyplot as plt


def load_data(filename):
    ''' 加载文本文件中的数据.
    '''
    dataset = []
    with open(filename, 'r') as f:
        for line in f:
            line_data = [float(data) for data in line.split()]
            dataset.append(line_data)
    return dataset

def split_dataset(dataset, feat_idx, value):
    ''' 根据给定的特征编号和特征值对数据集进行分割
    '''
    ldata, rdata = [], []
    for data in dataset:
        if data[feat_idx] < value:
            ldata.append(data)
        else:
            rdata.append(data)
    return ldata, rdata

def create_tree(dataset, fleaf, ferr, opt=None):
    ''' 递归创建树结构

    dataset: 待划分的数据集
    fleaf: 创建叶子节点的函数
    ferr: 计算数据误差的函数
    opt: 回归树参数.
        err_tolerance: 最小误差下降值;
        n_tolerance: 数据切分最小样本数
    '''
    if opt is None:
        opt = {'err_tolerance': 1, 'n_tolerance': 4}

    # 选择最优化分特征和特征值
    feat_idx, value = choose_best_feature(dataset, fleaf, ferr, opt)
    
    # 触底条件
    if feat_idx is None:
        return value

    # 创建回归树
    tree = {'feat_idx': feat_idx, 'feat_val': value}

    # 递归创建左子树和右子树
    ldata, rdata = split_dataset(dataset, feat_idx, value)
    ltree = create_tree(ldata, fleaf, ferr, opt)
    rtree = create_tree(rdata, fleaf, ferr, opt)
    tree['left'] = ltree
    tree['right'] = rtree

    return tree

def fleaf(dataset):
    ''' 计算给定数据的叶节点数值, 这里为均值
    '''
    dataset = np.array(dataset)
    return np.mean(dataset[:, -1])

def ferr(dataset):
    ''' 计算数据集的误差.
    '''
    dataset = np.array(dataset)
    m, _ = dataset.shape
    return np.var(dataset[:, -1])*dataset.shape[0]

def choose_best_feature(dataset, fleaf, ferr, opt):
    ''' 选取最佳分割特征和特征值

    dataset: 待划分的数据集
    fleaf: 创建叶子节点的函数
    ferr: 计算数据误差的函数
    opt: 回归树参数.
        err_tolerance: 最小误差下降值;
        n_tolerance: 数据切分最小样本数
    '''
    dataset = np.array(dataset)
    m, n = dataset.shape
    err_tolerance, n_tolerance = opt['err_tolerance'], opt['n_tolerance']

    err = ferr(dataset)
    best_feat_idx, best_feat_val, best_err = 0, 0, float('inf')

    # 遍历所有特征
    for feat_idx in range(n-1):
        values = dataset[:, feat_idx]
        # 遍历所有特征值
        for val in values:
            # 按照当前特征和特征值分割数据
            ldata, rdata = split_dataset(dataset.tolist(), feat_idx, val)
            if len(ldata) < n_tolerance or len(rdata) < n_tolerance:
                # 如果切分的样本量太小
                continue

            # 计算误差
            new_err = ferr(ldata) + ferr(rdata)
            if new_err < best_err:
                best_feat_idx = feat_idx
                best_feat_val = val
                best_err = new_err

    # 如果误差变化并不大归为一类
    if abs(err - best_err) < err_tolerance:
        return None, fleaf(dataset)

    # 检查分割样本量是不是太小
    ldata, rdata = split_dataset(dataset.tolist(), best_feat_idx, best_feat_val)
    if len(ldata) < n_tolerance or len(rdata) < n_tolerance:
        return None, fleaf(dataset)

    return best_feat_idx, best_feat_val

def get_nodes_edges(tree, root_node=None):
    ''' 返回树中所有节点和边
    '''
    Node = namedtuple('Node', ['id', 'label'])
    Edge = namedtuple('Edge', ['start', 'end'])

    nodes, edges = [], []

    if type(tree) is not dict:
        return nodes, edges

    if root_node is None:
        label = '{}: {}'.format(tree['feat_idx'], tree['feat_val'])
        root_node = Node._make([uuid.uuid4(), label])
        nodes.append(root_node)

    for sub_tree in (tree['left'], tree['right']):
        if type(sub_tree) is dict:
            node_label = '{}: {}'.format(sub_tree['feat_idx'], sub_tree['feat_val'])
        else:
            node_label = '{:.2f}'.format(sub_tree)
        sub_node = Node._make([uuid.uuid4(), node_label])
        nodes.append(sub_node)

        edge = Edge._make([root_node, sub_node])
        edges.append(edge)

        sub_nodes, sub_edges = get_nodes_edges(sub_tree, root_node=sub_node)
        nodes.extend(sub_nodes)
        edges.extend(sub_edges)

    return nodes, edges

def dotify(tree):
    ''' 获取树的Graphviz Dot文件的内容
    '''
    content = 'digraph decision_tree {\n'
    nodes, edges = get_nodes_edges(tree)

    for node in nodes:
        content += '    "{}" [label="{}"];\n'.format(node.id, node.label)

    for edge in edges:
        start, end = edge.start, edge.end
        content += '    "{}" -> "{}";\n'.format(start.id, end.id)
    content += '}'

    return content

def tree_predict(data, tree):
    ''' 根据给定的回归树预测数据值
    '''
    if type(tree) is not dict:
        return tree

    feat_idx, feat_val = tree['feat_idx'], tree['feat_val']
    if data[feat_idx] < feat_val:
        sub_tree = tree['left']
    else:
        sub_tree = tree['right']

    return tree_predict(data, sub_tree)

if '__main__' == __name__:
    datafile = 'ex0.txt'
    dataset = load_data(datafile)
    tree = create_tree(dataset, fleaf, ferr, opt={'n_tolerance': 4,
                                                  'err_tolerance': 1})

    dotfile = '{}.dot'.format(datafile.split('.')[0])
    with open(dotfile, 'w') as f:
        content = dotify(tree)
        f.write(content)

    dataset = np.array(dataset)
    # 绘制散点
    plt.scatter(dataset[:, 0], dataset[:, 1])
    # 绘制回归曲线
    x = np.linspace(0, 1, 50)
    y = [tree_predict([i], tree) for i in x]
    plt.plot(x, y, c='r')
    plt.show()

