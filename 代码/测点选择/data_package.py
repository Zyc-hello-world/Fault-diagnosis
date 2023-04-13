import numpy as np


def all_data_preprocess(data, method):
	pre_data = method(data)
	return pre_data

def splite_data_preprocess(data, method):
	i = 0
	for i in range(len(data) - 29):
		data[i: i+30] = method(data[i : i+30])
		i = i + 30
	return data


def Data2dAndLabels(data):
	"""
	data: 从文件中获取到的数据，（测点数*故障类型*样本数*采样点数）type = array
	
	return X: 二维的样本数据； Y:样本的标签【故障，测点】
	"""
	X = []
	Y = []
	for node_num in range(len(data)):
		for fault_num in range(len(data[node_num])):
			for sample in data[node_num][fault_num]:
				X.append(sample)
				Y.append([fault_num, node_num])

	return np.array(X), np.array(Y)



