import imp
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)

from typing import Counter
from keras import layers, losses
from keras import optimizers
from keras.layers.convolutional import Conv1D
from keras.optimizers import Adam
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn import svm


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE

import time

import copy
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from keras.utils import plot_model
import pydot
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from DataProcessing import Data_opt
from DataProcessing import Visualization
from DataProcessing import Run_method
from DataProcessing.ReadData import dataprocess
from MODEL.DBN import DBN
from MODEL.dbn_tf1.OPTS import DLOption

def runDBN(x_train, y_train, x_test, y_test, faults_name):
	print("-------DBN_model is running------------")
	tf.reset_default_graph()
	save_file = os.path.join(BASE_DIR, f"confusion_matrix/DBN_{circuit_name}.csv")
	opts = DLOption(5000, 0.001, 128, 0.9)

	dbn_model = DBN([70, 50], opts, faults, x_train, y_train, x_test, y_test)
	print("pre-train")
	dbn_model.pre_train()

	print("trainng........")
	dbn_model.train()
	print("OVER")
	y_pred, acc = dbn_model.predict(x_test, y_test)
	
	confusion_matrix = Visualization.vis_confusion_matrix(y_pred, y_test, faults_name, save_file)
	print(confusion_matrix)
	X = np.concatenate([x_train,x_test])
	Y = np.concatenate([y_train, y_test])
	print(X.shape, Y.shape)
	final_out = dbn_model.final_layer_out(X)
	# Visualization.vis_tsne(final_out, Y, faults_name)
	return acc


if __name__ == "__main__":
	print("------正在运testcd行DBN模型-----------")
	reference_index = 3
	is_norm = True
	data_info_dict = {"data_folder":["input/Sallen-Key/soft-fault50", 
                                  "input/Four-opamp/soft-fault50", 
                                  "input/leapfrog_filter/soft-fault50/",
                                  "input/ellipitic_filter/soft-fault50"], 
					"nodes" : [4, 8, 12, 14], 
					"faults" : [13, 13, 24, 34], 
					"mento" : [100, 200, 200, 200], 
					"features": [400, 200, 600, 500],
					# 时间单位为s
					"simulate_time": [0.00006, 0.0003, 0.001, 0.001],
     				"circuit_name": ["Sallen-Key", "Four-opamp", "leapfrog_filter", "ellipitic_filter"],
         			"out_node": [["V(1)","(V(4))"], ["V(2)","V(8)"], ["V(2)","V(10)"], ["V(10)","V(14)"]]}

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


	# 原始数据
	mydata = dataprocess(data_folder, nodes, faults, simulate_time, features, mento)
	mydata.get_data_from_folder()
	mydata.get_node_name()
	print("所有节点的名字为")
	print(mydata.nodes_name)

	print("所有故障的顺序为")
	print(mydata.fault_names)

	raw_data_with_label = mydata.raw_data[[mydata.nodes_name2num_dict[i] for i in stps_name]]
	raw_data = raw_data_with_label[:, :, :, :-1]
	stps_num = [mydata.nodes_name2num_dict[i] for i in stps_name]

	print(stps_num)

	print("源数据形状")
	print(raw_data_with_label.shape, raw_data.shape)
	x2, y2 = Data_opt.axis4_axis_2(raw_data_with_label, is_onehot=True)
	print(x2.shape, y2.shape)
	

	if is_norm:
		x2 = Data_opt.max_min_norm(x2)

	print("二维数据的格式为：{}-{}".format(x2.shape, y2.shape))
	# Visualization.vis_tsne(x2, y2, mydata.fault_names)
	
	# x_train ,y_train , x_test, y_test = Data_opt.get_split_2axis_data(x2, y2, len(stps_num), faults, mento, is_norm=True, test_ratio=0.5)
	# model.fit(x_train, np.argmax(y_train, axis=1).reshape((-1,1)))
	
	# train_score = model.score(x_train,np.argmax(y_train, axis=1).reshape(-1))
	# print("训练集：",train_score)
	# test_score = model.score(x_test, np.argmax(y_test, axis=1))
	# print("测试集：",test_score)
	
	"""
	#自己写的训练与测试集划分
	
	kfolds = 5
	test_acc = np.zeros(kfolds)
	for i in range(kfolds):
		x_train ,y_train , x_test, y_test = Data_opt.get_split_2axis_data(x2, y2, len(stps_num), faults, mento, test_ratio=0.3)
		test_acc[i] = runDBN(x_train, y_train, x_test, y_test, mydata.fault_names)
		print("第{}次的准确率为{}".format(i+1, test_acc[i]))
	plt.plot(test_acc)
	plt.show()
	print(test_acc)
	print("{}次测试的平均准确率为{}".format(kfolds, test_acc.mean()))
	"""
	Run_method.kfold_test(x2, y2, mydata.fault_names, runDBN)

	
