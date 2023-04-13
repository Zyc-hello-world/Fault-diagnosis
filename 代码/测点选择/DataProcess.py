import os 
import numpy as np 
import pandas as pd
import re

def get_fault_files_and_fault_names(intput_folder):
	fault_files = []
	fault_names = []
	if os.path.exists(intput_folder):
		files = os.listdir(intput_folder);
		for file in files:
			if file.split(".")[-1] == "csv":
				fault_files.append(file)
				name = file.split("/")[-1][:-4]
				fault_names.append(name)

	return fault_files, fault_names


def get_node_name(data, nodes):
	"""
	data：从一个文件中提取的数据
	nodes_name：测点的顺序列表，因为原始的测点并非从1-n
	"""
	nodes_name = []
	name = data.columns.values[1:]
	name = name.reshape((nodes, -1))
	for node in name[:,0]:
		# 注不同的文件其正则不一样，
		pattern = re.compile("(.*)@.*")
		seq = pattern.findall(node)
		nodes_name.append(seq[0])
	
	return nodes_name

def sample_for_data(data, time, samples):
	"""
	对data按时间进行采样，采样间隔为time/samples(秒)
	"""
		
	columns = [n.strip() for n in list(np.array(data.columns))]
	data.columns = columns
	new_data = pd.DataFrame(columns=columns)
	intval = time / samples
	cur = 0
	
	for i in range(1, samples+1):
		cur += intval
		cur = round(cur, 7)
			
		last_index = data.loc[data["Time"] <= cur].index[-1]

		next_index = data.loc[data["Time"] >= cur].index
		if len(next_index) != 0:
				
			cur_data = data.iloc[[last_index, next_index[0]],:].mean(axis=0)
		else:
			cur_data = data.iloc[last_index, :]
		new_data.loc[i-1] = cur_data

	return new_data



def get_data_from_file(file_path, nodes, mento, samples, label, time):
	"""从每一个文件读取数据,每个文件代表一类故障
	
	Args:
	    file_path (str): Description
	    nodes (int): Description
	    mento (int): Description
	    samples (int): Description
	    label (int): Description
	
	Returns:
	    TYPE: 	new_data (array(nodes, mento, samples+1)), 含有标签
	    		node_name (list[测点编号])
	"""
	data = pd.read_csv(file_path)
	# data = data.apply(lambda x : round(x, 6))
		
	node_name = get_node_name(data, nodes)
	#未打标签  data（二维数据） = (nodes_num * mento) * 采样点数

	new_data = sample_for_data(data, time, samples)

	new_data = new_data.iloc[:, 1:].values.T
	l = np.full(shape=nodes*mento, fill_value=label)
	
	
	l = l[:, np.newaxis]
	new_data = np.hstack((new_data, l))

	#将data 转为三维 nodes * mento * (samples+1)
	new_data = new_data.reshape((nodes, mento, -1))

	return new_data, node_name

def get_data_from_folder(data_folder, nodes, mento, samples, simulate_time):
	"""
	读取data_folder文件下的所有数据，将其转换为numpy格式，
		nodes：测点的数量
		samples：样本的数量
		return:四维数据（ 测点数*故障类型*样本数*采样点数）含标签
	"""

	fault_files, fault_names = get_fault_files_and_fault_names(data_folder)
	
	# datas = pd.DataFrame(columns=[])
	data_lists = []
	nodes_list = []

	for i, fault_file in enumerate(fault_files):

		file_path = os.path.join(data_folder, fault_file)
		data, nodes_name = get_data_from_file(file_path, nodes, mento, samples, i, simulate_time)
	
		data_lists.append(data)

	#将不同的采样数据置为相同的数量，以最小的为基准
	
	data = np.array(data_lists)

	data = np.swapaxes(data, 0, 1)

	return data, nodes_name, fault_names


def four_axis_to_two_axis(data, faults, nodes, mento):
	"""将四维数据转为二维带标签的array
	
	Args:
	    data (array): （node, faults, mento, samples+1）
	    faults (int): Description
	    nodes (int): Description
	    mento (int): Description
	
	Returns:
	    TYPE: 二维数据（nodes*faults*mento, samples+1）
	"""
	train_data = data.reshape((nodes*faults*mento, -1))

	return train_data

def get_node_select_data(data, node_number, faults, mento):
	"""由测点组合得到相应的数据
	
	Args:
	    data (TYPE): Description
	    node_number (list): 被选测点【0,1,2,5,...】
	    faults (TYPE): Description
	    mento (TYPE): Description
	
	Returns:
	    TYPE: 二维数组（）
	"""
	count = faults * mento
	part_data = []
	for i in node_number:
		part_data.append(data[i*count:i*count+count])

	part_data = np.array(part_data)
	print(part_data.shape)
	part_data = np.reshape(part_data, (len(node_number)*count, -1))

	return part_data
