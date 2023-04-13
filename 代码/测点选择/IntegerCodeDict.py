#!/usr/bin/env python
# coding: utf-8

# # 由混叠度矩阵来得到整数编码表

# In[5]:


import numpy as np
from UnionSet import Unionset


# In[1]:


def get_ICD(AliasMatrix_data, fault_num, node_num):
    """
    return: 整数编码表-fault_num*node_num
    """
    Threshold = 0
    ICD = np.zeros((fault_num, node_num))
    for node in range(node_num):
        temp = [0] * fault_num
        un = Unionset(fault_num)
        i = 0
        j = i + 1
        for k in AliasMatrix_data[node]:
            if j >= fault_num:
                i = i + 1
                j = i + 1
            if k != 0:
                un.join(i, j)
            j = j + 1
        for k in range(fault_num):
            
            a = un.parent[k]
            ICD[k][node] = un.parent[k]
    
    
    return ICD


# In[7]:


def get_FIlist(fault_table):
    """
    
    fault_table: 挑选出的测点集联合后构成的故障表。
    return: 隔离的故障编码
    
    """
    FI_list = []
    for fi in range(len(fault_table)):
        s = 0
        if len(fault_table[fi]) == 0:
            FI_list.append(fi)

    return FI_list

def get_fault_table(data, fault_num):
    """
    data:挑选出的测点联合构成的混叠表数据
    fault_num:故障数
    return：测点对应的每个故障极其混叠的故障列表, 混叠的故障对数，总的混叠度
            格式（测点*故障fi*和Fi混叠的故障编号
    """
    one_table = [[] for i in range(fault_num)]
    s = 0
    degree = 0
    i = 0
    alias_fault_pair = []
    for fi in range(fault_num):
        for fj in range(fi+1, fault_num):
            if data[i] != 0:
                s = s + 1
                degree = degree + data[i]
                one_table[fi].append(fj)
                one_table[fj].append(fi)
                alias_fault_pair.append([fi, fj, data[i]])
                
            i = i + 1
    if s != 0:
        degree = degree / s
    else:
        degree = 0
    return one_table, degree, alias_fault_pair


