#!/usr/bin/env python
# -*- coding: utf-8 -*-

from regression_tree import *

def not_tree(tree):
    ''' 判断是否不是一棵树结构
    '''
    return type(tree) is not dict

def collapse(tree):
    ''' 对一棵树进行塌陷处理, 得到给定树结构的平均值
    '''
    if not_tree(tree):
        return tree
    ltree, rtree = tree['left'], tree['right']
    return (collapse(ltree) + collapse(rtree))/2

def postprune(tree, test_data):
    ''' 根据测试数据对树结构进行后剪枝
    '''
    if not_tree(tree):
        return tree

    # 若没有测试数据则直接返回树平均值
    if not test_data:
        return collapse(tree)

    ltree, rtree = tree['left'], tree['right']

    if not_tree(ltree) and not_tree(rtree):
        # 分割数据用于测试
        ldata, rdata = split_dataset(test_data, tree['feat_idx'], tree['feat_val'])
        # 分别计算合并前和合并后的测试数据误差
        err_no_merge = (np.sum((np.array(ldata) - ltree)**2) +
                        np.sum((np.array(rdata) - rtree)**2))
        err_merge = np.sum((np.array(test_data) - (ltree + rtree)/2)**2)

        if err_merge < err_no_merge:
            print('merged')
            return (ltree + rtree)/2
        else:
            return tree

    tree['left'] = postprune(tree['left'], test_data)
    tree['right'] = postprune(tree['right'], test_data)

    return tree

