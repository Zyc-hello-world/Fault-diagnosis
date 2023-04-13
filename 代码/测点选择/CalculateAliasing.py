#!/usr/bin/env python
# coding: utf-8

# # 计算样本之间的混叠程度
# 使用谱聚类方法

# In[5]:


import numpy as np

def get_FaultPairData(feat_data, node_list):
    """由数据得到故障对
    
    Args:
        feat_data (四维数组): （nodes, faults, mento, samples）
        node_list (list): 测点编号
    
    Returns:
        list: [(fi, fj, [故障fi,fj共同构成的样本]),(),...]
    """
    fault_pair_data = []
    for node in feat_data:
        d = []
        for i in range(len(node)):
            for j in range(i+1, len(node)):

                c = node[i].tolist() + node[j].tolist()
                loc = (i, j, c)
                d.append(loc)
        fault_pair_data.append(d)
    return fault_pair_data


# In[1]:


def get_AliasMatrix(fault_pair_data, sample_num, save_theta, class_method, theta = 0):
    """
    class_method:传入分类的方法函数
    return:返回每个测点故障对的混叠度组成的二维矩阵 测点*故障对对应的混叠度
    """
    abnor_alias_pair = []
    node_alias = []
    i = 0
    for num, node in enumerate(fault_pair_data):
        fault_alias = []

        for fault_pair in node:
            
            i = i + 1
            pre_labels = class_method(fault_pair[2])
            
            first = sample_num + pre_labels[:sample_num].sum() - pre_labels[sample_num:].sum()
            second = sample_num - pre_labels[:sample_num].sum() + pre_labels[sample_num:].sum()
            alias_degree = min(first, second) / 2 / sample_num
            if alias_degree <= theta:
                alias_degree = 0
            fault_alias.append(alias_degree)
            if alias_degree >= save_theta:
                abnor_alias_pair.append((num, fault_pair[0], fault_pair[1], alias_degree))

        node_alias.append(fault_alias)
    return node_alias, abnor_alias_pair


# In[7]:


def merge_AliasM(node1_alias, node2_alias):
    """
    返回node1和node2的合并后的混叠度
    """
    
    al_deg = []
    
    for i in range(len(node1_alias)):
        min_val = min(node1_alias[i], node2_alias[i])
        al_deg.append(min_val)
    
    return al_deg

def merge(numbers, alias_data):

    al_deg = []
    for node in numbers:
        for i in range(len(alias_data[node])):
            min_val = min(node1_alias[i], node2_alias[i])
            al_deg.append(min_val)
    
    return al_deg
