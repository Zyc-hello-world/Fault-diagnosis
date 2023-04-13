import os
from pickle import FALSE
from pyexpat.model import XML_CQUANT_REP
import sys

from numpy.lib.shape_base import hsplit

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)
from typing import Counter
from keras import layers, losses
from keras import optimizers
from keras.layers.convolutional import Conv1D
from keras.optimizers import Adam
from sklearn import preprocessing
from sklearn.model_selection import KFold



import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE



import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from keras.utils import plot_model
import pydot

from DataProcessing import Data_opt
from DataProcessing import Visualization
from DataProcessing.ReadData import dataprocess

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,cohen_kappa_score
# 第一个电路的正则表达式为  ".*V\((\d+)\)@*"
# 第二个电路的正则表达式为  ".*V\(n\d+\)@*"
# 第三个电路的正则表达式为  ".*V\((.*)\)\)@.*" (V(OUT))@1    


if __name__ == "__main__":
	reference_index = 5
	is_norm = True
	print(BASE_DIR)
	data_info_dict = {"data_folder":["input/Sallen-Key/soft-fault50", 
                                  "input/Four-opamp/soft-fault50", 
                        			"input/leapfrog_filter/soft-fault50/", 
                           "input/ellipitic_filter/soft-fault50",
                           "input/KKCV-GA/caseII",
                           "input/leapfrog_filter/hard_fault"], 
					"nodes" : [4, 8, 12, 14, 14, 12], 
					"faults" : [13, 13, 23, 34, 45, 17], 
					"mento" : [100, 200, 200, 200, 50, 50], 
					"features": [400, 600, 600, 500, 50, 50],
					# 时间单位为s
					"simulate_time": [0.00008, 0.0003, 0.001, 0.005, 0.001, 0.001],
     				"circuit_name": ["Sallen-Key", "Four-opamp", "leapfrog_filter", "ellipitic_filter", "ellipitic_filter", "leapfrog_filter"],
         			"out_node": [["(V(4))"], ["V(8)"], ["V(12)"], 
                         ["V(2)","V(10)","V(14)"],
                        #  ["V(2)","V(3)","V(8)","V(9)","V(10)","(V(14))"],
                         ["V(1)","V(2)","V(3)","V(4)","V(5)","V(6)","V(7)","V(9)","V(10)","V(11)","V(12)","V(13)"],
                         ["V(n1)","V(n5)","V(n10)","V(n12)"]]}

	data_folder = data_info_dict["data_folder"][reference_index]
	nodes = data_info_dict["nodes"][reference_index]
	faults = data_info_dict["faults"][reference_index]
	mento = data_info_dict["mento"][reference_index]
	# 采样点数
	features = data_info_dict["features"][reference_index]
	simulate_time = data_info_dict["simulate_time"][reference_index]
	circuit_name = data_info_dict["circuit_name"][reference_index]
	stps_name = data_info_dict["out_node"][reference_index]
	print("选择的测点为:{}".format(stps_name))
	save_file_name = "+".join(stps_name)

	channels = len(stps_name)

	# 原始数据
	mydata = dataprocess(data_folder, nodes, faults, simulate_time, features, mento)
	mydata.get_data_from_folder()
	mydata.get_node_name()

	print("所有节点的名字为")
	print(mydata.nodes_name)

	print("所有故障的顺序为")
	print(mydata.fault_names)
	# ['C1_down', 'C1_up', 'C2_down', 'C2_up', 'normal', 'R2_down', 'R2_up', 'R3_down', 'R3_up', 'R4_down', 'R4_up', 'R5_down', 'R5_up']
	raw_data_with_label = mydata.raw_data[[mydata.nodes_name2num_dict[i] for i in stps_name]]
	raw_data = raw_data_with_label[:, :, :, :-1]
	stps_num = [mydata.nodes_name2num_dict[i] for i in stps_name]

	print(stps_num)
	print("源数据形状")
	print(raw_data_with_label.shape, raw_data.shape)

	x2, y2 = Data_opt.axis4_axis_2(raw_data_with_label, is_onehot=False)
	if is_norm:
		x2 = Data_opt.max_min_norm(x2)
	print("二维数据的格式为：{}-{}".format(x2.shape, y2.shape))
	kf = KFold(n_splits=5)
	for i in range(1):
		x_train ,y_train , x_test, y_test = Data_opt.get_split_2axis_data(x2, y2, len(stps_num), faults, mento, test_ratio=0.3)
	# i = 0
	# for k, (train_ind, test_ind) in enumerate(kf.split(x2)):
	# 	x_train = x2[train_ind]
	# 	x_test = x2[test_ind]
	# 	y_train = y2[train_ind]
	# 	y_test = y2[test_ind]
		clf = SVC(C=1.3)
		clf.fit(x_train, y_train)
		y_pred = clf.predict(x_test)
		print('使用SVM预测breast_cancer数据的准确率为：', accuracy_score(y_test,y_pred))      
		FP = 0
		FN = 0
		s = 0
		num = 
		print(num)
		for i in range(num):
			if y_test[i] == 14:
				s = s+1
				if y_pred[i] != 14:
					FP = FP + 1
			elif y_pred[i] == 14:
				FN = FN + 1
		print(s)
		print(f"虚警率为：{FP / num}")
		print(f"漏报率为：{FN / num}")
		# print('使用SVM预测breast_cancer数据的Cohen’s Kappa系数为：',cohen_kappa_score(y_test,y_pred))
	
