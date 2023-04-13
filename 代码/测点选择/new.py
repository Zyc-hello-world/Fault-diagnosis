import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)
import DataProcess
import FeatureExtraction
import CalculateAliasing
import GeneticAlgorithm
import GA_fitness
import save_data
from tsne import tsne
import km
import EM
import cal_deg
import MyKmeans
import search_all
import Visualize


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
import os
import time
import pandas as pd

reference_index = 1


data_info_dict = {"data_folder":["/input/data2/reference1/", "/input/KKCV-GA/caseII/", "/input/leapfrog_filter/hard_fault/",
                                "/input/Sallen-Key/soft-fault50/", "/input/Four-opamp/soft-fault50/", 
                                "/input/leapfrog_filter/soft-fault50/", "/input/ellipitic_filter/soft-fault50/"], 
                "nodes" : [11, 14, 12, 4, 8, 12, 14], 
                "faults" : [19, 45, 17, 13, 13, 24, 34], 
                "mento" : [30, 50, 50, 100, 200, 200, 200, 200], 
                "features": [50, 50, 50, 400, 600, 600, 500, 500],
                # 时间单位为s
                "simulate_time": [0.001, 0.001, 0.001, 0.00008, 0.0003, 0.001, 0.001],
                "circut_name": ["ref1", "case-II","hard", "Sallen-Key", "Four-opamp", "leapfrog_filter", "ellipitic_filter"]}

data_folder = BASE_DIR + data_info_dict["data_folder"][reference_index]
nodes = data_info_dict["nodes"][reference_index]
faults = data_info_dict["faults"][reference_index]
mento = data_info_dict["mento"][reference_index]
# 采样点数
samples = data_info_dict["features"][reference_index]
simulate_time = data_info_dict["simulate_time"][reference_index]
circuit_name = data_info_dict["circut_name"][reference_index]

print("-----------------------------")
print(f"当前的被测电路为{circuit_name}")
print("------------------------------")
n = 3           #小波包的分解层数
theta = 0.3
class_theta = 0     #设置混叠度阈值

pop_size = 20
max_iter = 50

# class_method = km.pre
class_method = EM.GMM
# class_method = MyKmeans.get_labels
class_method_name = class_method.__name__
print("正在使用的聚类算法{}".format(class_method_name))

# 原始数据

raw_4data_label, node_list, fault_name = DataProcess.get_data_from_folder(data_folder, nodes, mento, samples, simulate_time)
raw_4data = raw_4data_label[:,:,:,:-1]


raw_2data_label = DataProcess.four_axis_to_two_axis(raw_4data_label, faults, nodes, mento)
raw_2data = raw_2data_label[:, :-1]
label = raw_2data_label[:, -1]



# 我的特征提取,3层小波包提取能量，波动系数，...
myfeat_4data = FeatureExtraction.get_data(raw_4data, 3)
myfeat_2data = myfeat_4data.reshape((nodes*faults*mento, -1))

# print("myfeat_4data.shape={}, myfeat_2data.shape={}".format(myfeat_4data.shape, myfeat_2data.shape))

print(node_list)
print(fault_name)


def myfeat_LDA(myfeat_2data, label):
    myfeat_lda_2data_label = FeatureExtraction.MyLDA(myfeat_2data, label, 8)
    myfeat_lda_2data = myfeat_lda_2data_label[:, -1]

    myfeat_lda_4data_label = myfeat_lda_2data_label.reshape((nodes, faults, mento, -1))
    myfeat_lda_4data = myfeat_lda_4data_label[:, :, :, :-1]
    return myfeat_lda_4data
myfeat_lda_4data = myfeat_LDA(myfeat_2data, label)

file_GAresult = "GAresult.txt"
feat_pair_data = CalculateAliasing.get_FaultPairData(myfeat_lda_4data, node_list)




feat_alias_data, feat_abnormal_data = CalculateAliasing.get_AliasMatrix(feat_pair_data, mento, theta, class_method)

from CalculateAliasing import merge_AliasM

number = [0,2,3]
ren = [node_list[i] for i in number]
print(f"使用number为{number},  {ren}")
if len(number) != 0:
    cur_alias = (feat_alias_data[number[0]]).copy()
    for alias in number:
        cur_alias = merge_AliasM(cur_alias, feat_alias_data[alias])

print(len(cur_alias))

sum = 0
s = 0
for i in range(faults):
    for j in range(i+1, faults):
        if cur_alias[sum] != 0:
            print(f"{sum+1} {fault_name[i]} and {fault_name[j]} : {cur_alias[sum]}")
            s = s + cur_alias[sum]
        sum = sum + 1


print(s)
    