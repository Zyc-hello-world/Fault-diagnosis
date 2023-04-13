import os
import sys

from keras.callbacks import EarlyStopping

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

import time

import copy
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from keras.utils import plot_model
import pydot

from MODEL.dbn_tf1.OPTS import DLOption
from MODEL.conv1D import Conv1DNet
from DataProcessing import Data_opt
from DataProcessing import Visualization
from DataProcessing.ReadData import dataprocess
from keras.models import Model



def Run_CONV1_NET(x_train, y_train, x_test, y_test):
	print("-------------(以下为一维卷积的部分代码---------------")

	params = dict({"kernel":[7, 5, 3],
               "channel":[32, 64, 128],
               "FC":[100],
               "droporb":[0.4, 0.2, 0.2]})
	print(features, channels, faults)
	conv1_model = Conv1DNet(features, channels, faults, params)

	conv1_model.compile(optimizer=optimizers.Adam(0.0001),
						loss=losses.categorical_crossentropy,
						metrics=["accuracy"])
	# my_callback = [EarlyStopping(monitor="val_acc", patience=50, verbose=2, mode="max")]
	history = conv1_model.fit(x_train, y_train, 
					batch_size=128, epochs=300,
					# callbacks=my_callback,
					validation_data=(x_test, y_test))
	feature = Model(inputs=conv1_model.input, outputs=conv1_model.get_layer("FC0"))
	X = np.concatenate([x_train, x_test])
	Y = np.concatenate([y_train, y_test])
	final = feature.predict(X)
	Visualization.vis_tsne(X, Y, mydata.fault_names, f"{circuit_name}"-conv1D)
	if not os.path.exists("../model_visualize/conv1_model.png"):
		plot_model(conv1_model, to_file="../model_visualize/conv1_model.png", show_shapes=True)

	return conv1_model, history


def run_reference(x2, y2, circuit_name):
	kfolds = 5
	kf = KFold(n_splits=kfolds, shuffle=True)
	save_file = BASE_DIR + "/confusion_matrix/conv1_" + circuit_name + ".csv"
	test_acc = np.zeros(kfolds)
	print("------当前运行的模型是 一维卷积 -------")

#for i in range(kfolds):
	#x_train ,y_train , x_test, y_test = Data_opt.split_train_test(x2, y2, mento, test_ratio=0.3)
	i = 0
	for k, (train_ind, test_ind) in enumerate(kf.split(x2)):
		print(f"训练集的个数为{len(train_ind)}, 测试集的个数为{len(test_ind)}")
		x_train = x2[train_ind]
		x_test = x2[test_ind]
		y_train = y2[train_ind]
		y_test = y2[test_ind]
		print("第{}折验证".format(i))
		model, history = Run_CONV1_NET(x_train, y_train, x_test, y_test)
		print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
		test_acc[i] = history.history["val_accuracy"][-1]
		print(f"第{i}折测试的准确率为{test_acc}")
		i += 1

	y_pred = model.predict(x_test)
	confusion_matrix = Visualization.vis_confusion_matrix(y_pred, y_test, mydata.fault_names, save_file)
	train_acc = history.history["accuracy"]
	train_loss = history.history["loss"]
	val_acc = history.history["val_accuracy"]
	val_loss = history.history["val_loss"]
	Visualization.plot_model_results(train_acc, train_loss, val_acc, val_loss)
	print(confusion_matrix)
	plt.title("K-fold acc")
	plt.plot(test_acc)
	plt.show()
	return test_acc.mean()

if __name__ == "__main__":
	reference_index = 0
	is_norm = True
	data_info_dict = {"data_folder":["input/Sallen-Key/soft-fault50", 
                                  "input/Four-opamp/soft-fault50", 
                                  "input/leapfrog_filter/soft-fault50/",
                                  "input/ellipitic_filter/soft-fault50/"], 
					"nodes" : [4, 8, 12, 14], 
					"faults" : [13, 13, 23, 34], 
					"mento" : [100, 200, 200, 200], 
					"features": [400, 600, 600, 500],
					# 时间单位为s
					"simulate_time": [0.00008, 0.0003, 0.001, 0.005],
     				"circuit_name": ["Sallen-Key", "Four-opamp", "leapfrog_filter", "ellipitic_filter"],
         			"out_node": [["(V(4))"], ["V(8)"], ["V(2)","V(5)","V(10)","V(12)"], ["V(2)","V(10)","V(14)"]]}

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
	# raw_data_norm = Data_opt.max_min_norm(raw_data.reshape(len(stps_name)*faults*mento, -1))
	# raw_data_norm = raw_data_norm.reshape((len(stps_name), faults, mento, -1))
	stps_num = [mydata.nodes_name2num_dict[i] for i in stps_name]

	print(stps_num)

	print("源数据形状")
	print(raw_data_with_label.shape, raw_data.shape)

	
	x2, y2 = Data_opt.axis4_axis_2(raw_data_with_label, is_onehot=True)
	x3, y3 = Data_opt.axis4_axis3(raw_data_with_label, is_onehot=True)
	x3_change = np.swapaxes(x3, 1, 2)
	if is_norm:
		x3_norm = x3_change.reshape((-1, features))
		x3_norm = Data_opt.max_min_norm(x3_norm)
	x3_norm = x3_norm.reshape((-1, len(stps_num), features))
	x3_norm = np.swapaxes(x3_norm, 1, 2)
	if is_norm:
		x2 = Data_opt.max_min_norm(x2)
	x2 = x2[:, :, np.newaxis]

	
	print("需要的数据格式为：{}".format(x2.shape, y2.shape))
	print(f"需要的数据格式为: {x3.shape}, {y3.shape}")
	
	test_acc = run_reference(x3_norm, y3, circuit_name)
	print("平均准确率为:{}".format(test_acc))
	
