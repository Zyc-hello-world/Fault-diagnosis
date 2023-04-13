

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
import os
import time
import Data_opt
import copy
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from keras.utils import plot_model
import pydot


from Data_opt import *
import Visualization
from DataProcess import dataprocess
from Data2Image import get_image
from ResNet50 import ResNet50
"""

	data: 					从文件中获取到的数据，（测点数*故障类型*样本数*采样点数）type = array


							每个故障对形如（i, j, pair_data），i,j为故障编号，pair_data为这两类故障的所有样本，i类在前


	node_list:				节点的编号顺序
	fault_name:				故障的名字

"""


if __name__ == "__main__":
	reference_index = 0

	data_info_dict = {"data_folder":["../input/Sallen-Key/soft-fault30", "../input/Four-opamp/soft-fault30", "../input/leapfrog_filter/soft-fault30/"], 
					"nodes" : [4, 8, 12], 
					"faults" : [13, 13, 21], 
					"mento" : [200, 200, 200], 
					"features": [400, 600, 600],
					# 时间单位为s
					"simulate_time": [0.00008, 0.0003, 0.001],
     				"circuit_name": ["Sallen-Key", "Four-opamp", "leapfrog_filter"],
         			"out_node": [["V(4)"], ["V(8)"], ["V(12)"]]}

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

	print("-------------以下为ResNet50的部分代码---------------")
	get_image(raw_data, mydata.nodes_name2num_dict,"./images/"+circuit_name )
	kfolds = 5
	test_acc = np.zeros(kfolds)
	model = ResNet50(include_top=True, weights=None, classes=17)

	model.compile(loss="categorical_crossentropy",
				optimizer=optimizers.Adam(0.001), 
				metrics=['accuracy'])
	model.summary()
	train_gen = read_img("./images/leapfrog_filter/train/")
	test_gen = read_img("./images/leapfrog_filter/test/")
	history = model.fit_generator(train_gen, epochs=100, validation_data=test_gen)
	
		