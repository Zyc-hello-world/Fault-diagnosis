import os 
import numpy as np 
import pandas as pd
import re
import sys
print(sys.path)

class dataprocess():
	def __init__(self, input_folder, nodes, faults, time, samples, mento):
		self.input_folder = input_folder
		self.nodes = nodes
		self.faults = faults
		self.time = time 
		self.samples = samples
		self.mento = mento
		self.fault_names = []
		self.nodes_name = []
		self.nodes_name2num_dict = {}
		self.raw_data = None
		self.faults_name2num_dict = {}
		self.columns = []


	def get_fault_files_and_fault_names(self):
		fault_files = []
		if os.path.exists(self.input_folder):
			files = os.listdir(self.input_folder)

			for file in files:

				if file.split(".")[-1] == "csv":
					fault_files.append(file)
					name = file.split("/")[-1][:-4]
					self.fault_names.append(name)
					self.faults_name2num_dict[name] = len(self.faults_name2num_dict)
		
		return fault_files


	def get_node_name(self):
		"""
		data：从一个文件中提取的数据
		nodes_name：测点的顺序列表，因为原始的测点并非从1-n
		"""
		nodes_name = []
		
		name = self.columns.reshape((self.nodes, -1))
		for node in name[:,0]:
			# 注不同的文件其正则不一样，
			pattern = re.compile("(.*)@.*")
			seq = pattern.findall(node)
			nodes_name.append(seq[0].strip())
		self.nodes_name = nodes_name

		for num, name in enumerate(self.nodes_name):
			self.nodes_name2num_dict[name] = num

	def sample_for_data(self, data):
		"""
		对data按时间进行采样，采样间隔为time/samples(秒)
		"""
		
		columns = [n.strip() for n in list(np.array(data.columns))]
		data.columns = columns
		new_data = pd.DataFrame(columns=columns)
		intval = self.time / self.samples
		cur = 0
	
		for i in range(1, self.samples+1):
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



	def get_data_from_file(self, file_path, label):
		"""从每一个文件读取数据,每个文件代表一类故障
		
		Args:
		    file_path (str): Description
		    label (int): Description
		
		Returns:
		    TYPE: 	new_data (array(nodes, mento, samples+1)), 含有标签
		    		node_name (list[测点编号])
		"""
		data = pd.read_csv(file_path)
		self.columns = data.columns.values[1:]
		# data = data.apply(lambda x : round(x, 6))
			
		#未打标签  data（二维数据） = (nodes_num * mento) * 采样点数

		new_data = self.sample_for_data(data)

		new_data = new_data.iloc[:, 1:].values.T
		l = np.full(shape=self.nodes * self.mento, fill_value=label)
		
		
		l = l[:, np.newaxis]
		new_data = np.hstack((new_data, l))

		#将data 转为三维 nodes * mento * (samples+1)
		new_data = new_data.reshape((self.nodes, self.mento, -1))

		return new_data

	def get_data_from_folder(self):
		"""
		读取data_folder文件下的所有数据，将其转换为numpy格式，
			nodes：测点的数量
			samples：样本的数量
			return:四维数据（ 测点数*故障类型*样本数*采样点数）含标签
		"""
		print("--------")
		fault_files = self.get_fault_files_and_fault_names()
		
		# datas = pd.DataFrame(columns=[])
		data_lists = []
		
		for i, fault_file in enumerate(fault_files):
			
			file_path = os.path.join(self.input_folder, fault_file)
			
			data = self.get_data_from_file(file_path , i)
		
			data_lists.append(data)

		#将不同的采样数据置为相同的数量，以最小的为基准
		
		raw_data = np.array(data_lists)
		raw_data = np.swapaxes(raw_data, 0, 1)
		self.raw_data = raw_data.copy()


	def four_axis_to_two_axis(self, raw_data, nodes):
		"""将四维数据转为二维带标签的array
		
		Args:
		    data (array): （node, faults, mento, samples+1）

		    nodes (int): Description

		
		Returns:
		    TYPE: 二维数据（nodes*faults*mento, samples+1）
		"""
		train_data = raw_data.reshape((nodes * self.faults * self.mento, -1))

		return train_data



	def get_node_select_data(self, data, node_number):
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
