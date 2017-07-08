#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: PytLab <shaozhengjiang@gmail.com>
# Date: 2017-07-07

import copy
import uuid
import pickle
from collections import defaultdict, namedtuple
from math import log2


class DecisionTreeClassifier(object):
    ''' 使用ID3算法划分数据集的决策树分类器
    '''

    @staticmethod
    def split_dataset(dataset, classes, feat_idx):
        ''' 根据某个特征以及特征值划分数据集

        :param dataset: 待划分的数据集, 有数据向量组成的列表.
        :param classes: 数据集对应的类型, 与数据集有相同的长度
        :param feat_idx: 特征在特征向量中的索引

        :param splited_dict: 保存分割后数据的字典 特征值: [子数据集, 子类型列表]
        '''
        splited_dict = {}
        for data_vect, cls in zip(dataset, classes):
            feat_val = data_vect[feat_idx]
            sub_dataset, sub_classes = splited_dict.setdefault(feat_val, [[], []])
            sub_dataset.append(data_vect[: feat_idx] + data_vect[feat_idx+1: ])
            sub_classes.append(cls)

        return splited_dict

    def get_shanno_entropy(self, values):
        ''' 根据给定列表中的值计算其Shanno Entropy
        '''
        uniq_vals = set(values)
        val_nums = {key: values.count(key) for key in uniq_vals}
        probs = [v/len(values) for k, v in val_nums.items()]
        entropy = sum([-prob*log2(prob) for prob in probs])
        return entropy

    def choose_best_split_feature(self, dataset, classes):
        ''' 根据信息增益确定最好的划分数据的特征

        :param dataset: 待划分的数据集
        :param classes: 数据集对应的类型

        :return: 划分数据的增益最大的属性索引
        '''
        base_entropy = self.get_shanno_entropy(classes)

        feat_num = len(dataset[0])
        entropy_gains = []
        for i in range(feat_num):
            splited_dict = self.split_dataset(dataset, classes, i)
            new_entropy = sum([
                len(sub_classes)/len(classes)*self.get_shanno_entropy(sub_classes)
                for _, (_, sub_classes) in splited_dict.items()
            ])
            entropy_gains.append(base_entropy - new_entropy)

        return entropy_gains.index(max(entropy_gains))

    def get_majority(classes):
        ''' 返回类型中占据大多数的类型
        '''
        cls_num = defaultdict(lambda: 0)
        for cls in classes:
            cls_num[cls] += 1

        return max(cls_num)

    def create_tree(self, dataset, classes, feat_names):
        ''' 根据当前数据集递归创建决策树

        :param dataset: 数据集
        :param feat_names: 数据集中数据相应的特征名称
        :param classes: 数据集中数据相应的类型

        :param tree: 以字典形式返回决策树
        '''
        # 如果数据集中只有一种类型停止树分裂
        if len(set(classes)) == 1:
            return classes[0]

        # 如果遍历完所有特征，返回比例最多的类型
        if len(feat_names) == 0:
            return get_majority(classes)

        # 分裂创建新的子树
        tree = {}
        best_feat_idx = self.choose_best_split_feature(dataset, classes)
        feature = feat_names[best_feat_idx]
        tree[feature] = {}

        # 创建用于递归创建子树的子数据集
        sub_feat_names = feat_names[:]
        sub_feat_names.pop(best_feat_idx)

        splited_dict = self.split_dataset(dataset, classes, best_feat_idx)
        for feat_val, (sub_dataset, sub_classes) in splited_dict.items():
            tree[feature][feat_val] = self.create_tree(sub_dataset,
                                                       sub_classes,
                                                       sub_feat_names)
        self.tree = tree
        self.feat_names = feat_names

        return tree

    def get_nodes_edges(self, tree=None, root_node=None):
        ''' 返回树中所有节点和边
        '''
        Node = namedtuple('Node', ['id', 'label'])
        Edge = namedtuple('Edge', ['start', 'end', 'label'])

        if tree is None:
            tree = self.tree

        if type(tree) is not dict:
            return [], []

        nodes, edges = [], []

        if root_node is None:
            label = list(tree.keys())[0]
            root_node = Node._make([uuid.uuid4(), label])
            nodes.append(root_node)

        for edge_label, sub_tree in tree[root_node.label].items():
            node_label = list(sub_tree.keys())[0] if type(sub_tree) is dict else sub_tree
            sub_node = Node._make([uuid.uuid4(), node_label])
            nodes.append(sub_node)

            edge = Edge._make([root_node, sub_node, edge_label])
            edges.append(edge)

            sub_nodes, sub_edges = self.get_nodes_edges(sub_tree, root_node=sub_node)
            nodes.extend(sub_nodes)
            edges.extend(sub_edges)

        return nodes, edges

    def dotify(self, tree=None):
        ''' 获取树的Graphviz Dot文件的内容
        '''
        if tree is None:
            tree = self.tree

        content = 'digraph decision_tree {\n'
        nodes, edges = self.get_nodes_edges(tree)

        for node in nodes:
            content += '    "{}" [label="{}"];\n'.format(node.id, node.label)

        for edge in edges:
            start, label, end = edge.start, edge.label, edge.end
            content += '    "{}" -> "{}" [label="{}"];\n'.format(start.id, end.id, label)
        content += '}'

        return content

    def classify(self, data_vect, feat_names=None, tree=None):
        ''' 根据构建的决策树对数据进行分类
        '''
        if tree is None:
            tree = self.tree

        if feat_names is None:
            feat_names = self.feat_names

        # Recursive base case.
        if type(tree) is not dict:
            return tree

        feature = list(tree.keys())[0]
        value = data_vect[feat_names.index(feature)]
        sub_tree = tree[feature][value]

        return self.classify(feat_names, data_vect, sub_tree)

    def dump_tree(self, filename, tree=None):
        ''' 存储决策树
        '''
        if tree is None:
            tree = self.tree

        with open(filename, 'w') as f:
            pickle.dump(tree, f)

    def load_tree(self, filename):
        ''' 加载树结构
        '''
        with open(filename, 'r') as f:
            tree = pickle.load(f)
            self.tree = tree
        return tree

