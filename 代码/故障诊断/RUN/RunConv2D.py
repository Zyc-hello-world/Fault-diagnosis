import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)

from typing import Counter
from keras import layers, losses
from keras import optimizers
from keras.layers.convolutional import Conv1D, Conv2D
from keras.optimizers import Adam
from sklearn import preprocessing
from sklearn.model_selection import KFold



import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
import os
import time

import copy
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from keras.utils import plot_model
import pydot

from DataProcessing import Data_opt, Signal_preprocessing
from DataProcessing import Visualization
from DataProcessing.ReadData import dataprocess
from MODEL import conv2D

"""

	data: 					从文件中获取到的数据，（测点数*故障类型*样本数*采样点数）type = array


							每个故障对形如（i, j, pair_data），i,j为故障编号，pair_data为这两类故障的所有样本，i类在前


	node_list:				节点的编号顺序
	fault_name:				故障的名字

"""



# print("----------多尺度卷积 is runnng----------")

def train(x_train, y_train, x_test, y_test):
	save_file = "../model_visualize/Conv2D_"+circuit_name+".png"
	features = x_train.shape[2]
	freqs = x_train.shape[1]
	model = conv2D.MultiScale(freqs, features,len(stps_name), faults)
	Visualization.save_model_fig(model, save_file)
	model.summary()
	model.compile(optimizer=optimizers.adam(0.0001),
				loss=losses.categorical_crossentropy,
				metrics=["accuracy"])
	print(x_train.shape)
	history = model.fit(x_train, y_train, 
					 batch_size=128, epochs=500,
			validation_data=(x_test, y_test))
	return model, history



def run(all_data, circuit_name):
	kfolds = 1
	test_acc = np.zeros(kfolds)
	x3_norm = all_data["x3_norm"]
	x3 = all_data["x3"]
	y3 = all_data["y3"]
	save_file="../confusion_matrix/Conv2D-"+circuit_name+".csv"
	for i in range(kfolds):
		x_train, y_train, x_test, y_test = Data_opt.get_split_3axis_data(x3, y3, len(stps_num), faults, mento, test_ratio=0.3)
	
	# kf = KFold(n_splits=5)
	# i = 0
	# for train_ind, test_ind in kf.split(x3_norm):
		print("第{}折验证".format(i))
		# x_train = x3_norm[train_ind]
		# x_test = x3_norm[test_ind]
		# y_train = y3[train_ind]
		# y_test = y3[test_ind]
		model, history = train(x_train, y_train, x_test, y_test)

		acc = history.history["accuracy"]
		loss = history.history["loss"]
		val_acc = history.history["val_accuracy"]
		val_loss = history.history["val_loss"]
		
		Visualization.plot_model_results(acc, loss, val_acc, val_loss)
		test_acc[i] = history.history["val_accuracy"][-1]

		# wpt_test = get_wide_data(x_test, n=5)[:,:,np.newaxis]
		y_pred = model.predict(x_test)
		print("测试准确率为{}".format(test_acc))
	confusion_matrix = Visualization.vis_confusion_matrix(y_pred, y_test, mydata.fault_names, save_file)
	print(confusion_matrix)
	return test_acc.mean()

if __name__ == "__main__":
	print("当前运行的模型是多尺度卷积的模型")
	reference_index = 3
	is_norm = True

	data_info_dict = {"data_folder":["../input/Sallen-Key/soft-fault50", 
								  "../input/Four-opamp/soft-fault50", 
								  "../input/leapfrog_filter/soft-fault50/",
								  "../input/ellipitic_filter/soft-fault50"], 
					"nodes" : [4, 8, 12, 14], 
					"faults" : [13, 13, 23, 34], 
					"mento" : [100, 200, 200, 200], 
					"features": [400, 600, 600, 500],
					# 时间单位为s
					"simulate_time": [0.00008, 0.0003, 0.001, 0.005],
	 				"circuit_name": ["Sallen-Key", "Four-opamp", "leapfrog_filter", "ellipitic_filter"],
		 			"out_node": [["(V(4))"], ["V(8)"], ["V(12)"], ["V(14)"]]}

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

	raw_data_with_label = mydata.raw_data[[mydata.nodes_name2num_dict[i] for i in stps_name]]
	raw_data = raw_data_with_label[:, :, :, :-1]
	stps_num = [mydata.nodes_name2num_dict[i] for i in stps_name]

	print(stps_num)

	print("源数据形状")
	print(raw_data_with_label.shape, raw_data.shape)

	
	x3, y3 = Data_opt.axis4_axis3(raw_data_with_label, is_onehot=True)
	if is_norm:
		x3_norm = x3.reshape((-1, features))
		x3_norm = Data_opt.max_min_norm(x3_norm)
	x3_norm = x3_norm.reshape((-1, features, len(stps_name)))
	x3 = Signal_preprocessing.get_wpt_data(x3, 5)
	print(x3.shape)
	all_data = {"x3": x3,
			 	"y3": y3,
			  	"x3_norm" : x3_norm}
	print("三维数据格式为：{}-{}, {}".format(x3.shape, y3.shape, x3_norm.shape))
	

	test_acc = run(all_data, circuit_name)
	print("平均准确率为:{}".format(test_acc))
