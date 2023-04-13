#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pywt
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import re

from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# # 提取分解系数

# In[7]:


def wpt_plt(signal,n):
    #wpt分解
    wp = pywt.WaveletPacket(data=signal, wavelet='db1',mode='symmetric',maxlevel=n)
 
    #计算每一个节点的系数，存在map中，key为'aa'等，value为列表
    data_map = {}
    data_map[1] = signal
    for row in range(1,n+1):
        lev = []
        for i in [node.path for node in wp.get_level(row, 'freq')]:
            data_map[i] = wp[i].data
 
    #作图
    plt.figure(figsize=(15, 10))
    plt.subplot(n+1,1,1) #绘制第一个图
    plt.plot(data_map[1])
    for i in range(2,n+2):
        level_num = pow(2,i-1)  #从第二行图开始，计算上一行图的2的幂次方
        #获取每一层分解的node：比如第三层['aaa', 'aad', 'add', 'ada', 'dda', 'ddd', 'dad', 'daa']
        re = [node.path for node in wp.get_level(i-1, 'freq')]  
        for j in range(1,level_num+1):
            plt.subplot(n+1,level_num,level_num*(i-1)+j)
            plt.plot(data_map[re[j-1]]) #列表从0开始
    
    return data_map


# In[9]:


def calculate_energy(n, wpt_data):
    """
    n：bin示小波包的分解层数
    data_map:由小波包的分解按频率顺序构成的数据列表 
    """
    #第n层能量特征
    energy = []
    for data in wpt_data:
        energy.append(pow(np.linalg.norm(data,ord=None),2))
    return np.array(energy)


# In[10]:


def calculate_fluctuation(n, wpt_data):
    
    fluctuation_coefficients = []
    for data in wpt_data:
        sum = 0
        for m in range(len(data)-1):
            sum = sum + (data[m+1] - data[m])
        fluctuation_coefficients.append(sum / (len(data)-1))
    return np.array(fluctuation_coefficients)


# In[20]:


def calculate_skewness(n, wpt_data):
    
    skewness = []
    
    for data in wpt_data:
        sk = 0
        ave = np.mean(data)
        molecular = 0
        Denominator = 0
        for c in data:
            molecular = molecular + pow(c - ave,3)
            Denominator = Denominator + pow(c - ave, 2)
        molecular = molecular / len(data)
        Denominator = pow(Denominator / len(data), 3/2)
        
        skewness.append(molecular / Denominator)
        
    return np.array(skewness)
            


# In[1]:


def get_aver_feat(data, n): 
    """
    求平均特征
    data : 原始的四维数据 type = array
    n : 小波包分解的层数
    ave_data : 返回的平均后的数据（测点*故障类型*特征数*小波包分解最后一层数）  type = array
    """
    
    ave_data = []
    for node in data:
        ave_fault = []
        for fault in node:
            ave_energy = np.zeros(pow(2, n))
            ave_fluctuation_coefficient = np.zeros(pow(2, n))
            ave_skewness = np.zeros(pow(2, n))
            sample_num = fault[0].size
            for sample in fault:
                wpt_data = []
                
                wp = pywt.WaveletPacket(data=sample, wavelet='db1',mode='symmetric',maxlevel=3)
                for i in [node.path for node in wp.get_level(n, 'freq')]:
                    wpt_data.append(wp[i].data)
                
                ave_energy = ave_energy + calculate_energy(n, wpt_data)
                ave_fluctuation_coefficient = ave_fluctuation_coefficient + calculate_fluctuation(n, wpt_data)
                ave_skewness = ave_skewness + calculate_skewness(n, wpt_data)
            
            ave_energy = ave_energy / sample_num
            ave_fluctuation_coefficiente = ave_fluctuation_coefficient / samples_num
            ave_skewness = ave_energy / sample_num
        
            a = np.vstack((ave_energy, ave_fluctuation_coefficient))
            b = np.vstack((a, ave_skewness))
            ave_fault.append(b)
        
        ave_fault = np.array(ave_fault)
        ave_data.append(ave_fault)
    return np.array(ave_data)


# In[2]:


def get_data(data, n):
    """
    data:原始数据
    n:小波包分解层数
    return: 需要举类的数据结构（测点*故障*样本*特征）
    """
    new_data = []
    for node in data:
        fault_data = []
        for fault in node:
            sample_feat = []
            for sample in fault:
                wpt_data = []
                
                wp = pywt.WaveletPacket(data=sample, wavelet='db1',mode='symmetric',maxlevel=n)
                for i in [node.path for node in wp.get_level(n, 'freq')]:
                    wpt_data.append(wp[i].data)
                
                energy = calculate_energy(n, wpt_data)
                fluctuation_coefficient =  calculate_fluctuation(n, wpt_data)
                skewness = calculate_skewness(n, wpt_data)
        
                a = np.hstack((energy, fluctuation_coefficient))
                b = np.hstack((a, skewness))
                #将特征中小于0.1的值置为0
                # b = [x if abs(x) > 0.1 else 0 for x in b]
                sample_feat.append(b)
            sample_feat = np.array(sample_feat)
            fault_data.append(sample_feat)
        fault_data = np.array(fault_data)
        new_data.append(fault_data)
    return np.array(new_data)

def get_ref1_featdata(data, n):
    """
    获得参考文献1的特征数据，即5层小波包分解的数据，
    """
    feat_data = []
    for node in data:
        fault_data = []
        for fault in node:
            sample_feat = []
            for sample in fault:
                energy = []
                
                wp = pywt.WaveletPacket(data=sample, wavelet='db1',mode='symmetric',maxlevel=n)
                for i in [node.path for node in wp.get_level(n, 'freq')]:
                    energy.append(pow(np.linalg.norm(wp[i].data ,ord=None),2))
                sample_feat.append(energy)
            fault_data.append(sample_feat)
        feat_data.append(fault_data)

    feat_data = np.array(feat_data)
    return feat_data

def Pca(energy_data, n_components):

    # 进行PCA分析， 0.006
    
    zscore = preprocessing.StandardScaler()
    
    
    node_num = energy_data.shape[0]
    fault_num = energy_data.shape[1]
    sample_num = energy_data.shape[2]


    energy = energy_data.reshape((node_num * fault_num * sample_num, -1))
    nor_energy = zscore.fit_transform(energy)
    
    pca = PCA(n_components=n_components)
    pca_data = pca.fit(nor_energy).transform(nor_energy)

    ratios = pca.explained_variance_ratio_

    return pca_data


def MyLDA(data, label, n_components):
    """LDA降维
    
    Args:
        data (二维数组): （nodes*faults*mento, samples+1）
    
    Returns:
        TYPE: 降维后的数据（nodes*faults*mento, n_components）含标签
    """
    print("LDA")

    print(data.shape)
    
    lda = LDA(n_components=n_components).fit_transform(data, label)
    label = label[:, np.newaxis]
    print(lda.shape, label.shape)
    lda_data = np.hstack([lda, label])
    print(lda_data.shape)
    return lda_data