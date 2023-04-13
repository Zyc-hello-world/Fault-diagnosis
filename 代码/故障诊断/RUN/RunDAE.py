import os
from pickle import FALSE
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
from MODEL.DAE import DAE

"""

	data: 					从文件中获取到的数据，（测点数*故障类型*样本数*采样点数）type = array


							每个故障对形如（i, j, pair_data），i,j为故障编号，pair_data为这两类故障的所有样本，i类在前


	node_list:				节点的编号顺序
	fault_name:				故障的名字

"""
"""
50%的故障，共11个
电容电阻5%容差
"""
# 第一个电路的正则表达式为  ".*V\((\d+)\)@*"
# 第二个电路的正则表达式为  ".*V\(n\d+\)@*"
# 第三个电路的正则表达式为  ".*V\((.*)\)\)@.*" (V(OUT))@1    


def Run_DAE_NET(x_train, y_train, x_test, y_test):
	print("-------------以下为DAE(降噪自动编码机)的部分代码---------------")
	

	features = x_train.shape[1]
	lambda1 = 4
	lambda2 = 5
	sizes = [features,200, 100, 50, faults]
	dae_model = DAE(sizes)
	dae_model.compile(optimizer=optimizers.adam(0.0001),
				  loss={"decode":losses.mse, "class":losses.categorical_crossentropy},
				  loss_weights={"decode":lambda1, "class":lambda2},
				  metrics=["accuracy"])

	history = dae_model.fit(x_train, [x_train, y_train], 
			  batch_size=128, epochs=1000, shuffle=True,
			  validation_data=(x_test, [x_test, y_test]))
	if not os.path.exists("../model_visualize/dae_model.png"):
		plot_model(dae_model, to_file='../model_visualize/dae_model.png',show_shapes=True)
	return dae_model, history



def run_reference(x2, y2, faults_name, circuit_name, kfolds=1):
	test_acc = np.zeros(kfolds)
	print("------当前运行的模型是 DAE------")
	save_file = BASE_DIR + "../confusion_matrix/DAE_" + circuit_name + ".csv"
	kf = KFold(n_splits=5)
	for k, (train_ind, test_ind) in enumerate(kf.split(x2)):
#for i in range(kfolds):
#x_train ,y_train , x_test, y_test = Data_opt.get_split_2axis_data(x2, y2, len(stps_num), faults, mento, test_ratio=0.3)
	# i = 0
	# for train_ind, test_ind in kf.split(x2):
		x_train = x2[train_ind]
		x_test = x2[test_ind]
		y_train = y2[train_ind]
		y_test = y2[test_ind]
		model, history = Run_DAE_NET(x_train, y_train, x_test, y_test)
#print(history.history.keys())
#print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
		
		test_acc[i] = history.history["val_class_acc"][-1]
		print("第{}折的准确率为{}".format(k+1, test_acc[k]))
		# i += 1
	y_pred = model.predict(x_test)[1]

	confusion_matrix = Visualization.vis_confusion_matrix(y_pred, y_test, mydata.fault_names, save_file)
	train_acc = history.history["class_accuracy"]
	train_loss = history.history["class_loss"]
	val_acc = history.history["val_class_accuracy"]
	val_loss = history.history["val_class_loss"]
	Visualization.plot_model_results(train_acc, train_loss, val_acc, val_loss)
	plt.title("K-fold acc")
	plt.plot(test_acc)
	plt.show()
	print(confusion_matrix)
	print(test_acc)
	return test_acc.mean()

if __name__ == "__main__":
	reference_index = 2
	is_norm = True
	print(BASE_DIR)
	data_info_dict = {"data_folder":["input/Sallen-Key/soft-fault50", 
                                  "input/Four-opamp/soft-fault50", 
                        			"input/leapfrog_filter/soft-fault50/", 
                           "input/ellipitic_filter/soft-fault50"], 
					"nodes" : [4, 8, 12, 14], 
					"faults" : [13, 13, 23, 34], 
					"mento" : [100, 200, 200, 200], 
					"features": [400, 600, 600, 500],
					# 时间单位为s
					"simulate_time": [0.00008, 0.0003, 0.001, 0.005],
     				"circuit_name": ["Sallen-Key", "Four-opamp", "leapfrog_filter", "ellipitic_filter"],
         			"out_node": [["(V(4))"], ["V(8)"], ["V(12)"], ["V(2)","V(10)","V(14)"]]}

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

	x2, y2 = Data_opt.axis4_axis_2(raw_data_with_label, is_onehot=True)
	if is_norm:
		x2 = Data_opt.max_min_norm(x2)
	print("二维数据的格式为：{}-{}".format(x2.shape, y2.shape))
	

	test_acc = run_reference(x2, y2, mydata.fault_names, circuit_name)
	print("该模型的准确率为:{}".format(test_acc))
	
	
