import numpy as np
import pandas as pd

def save_alias_data(filename, alias_data):
	"""
	保存混叠表
	"""

	with open(filename, 'w+', encoding='utf-8') as f:
		for line in alias_data:
			for num in line:
				f.write(str(num))
				f.write(" ")
			f.write("\n")

def save_alias_dataToCsv(alias_data, faults):

	for node, line in enumerate(alias_data):
		data = np.zeros((faults, faults))
		sum = 0
		for i in range(faults):
			for j in range(i+1, faults):
				data[i][j] = line[sum]
				sum = sum + 1
		np.savetxt(f"E:/AllCode/Python/subject/py/table/node{node}.csv", data, delimiter=",")
		print(f"have sucessful writ node{node}")

def load_alias_data(filename):
	"""
	加载混叠表
	"""
	with open(filename, "r") as f:
		lines = f.readlines()
		alias_matrix = []
		for line in lines:
			line = list(line.strip().split(" "))
			alias_matrix.append(np.float64(line))

	return alias_matrix

def save_abnormal_data(filename, abnormal_data):
	"""
	保存混叠度较大的故障对,以元组形式存储
	"""
	with open(filename, "w+", encoding="utf-8") as f:
		for pair in abnormal_data:
			for one in pair:
				f.write(str(one))
				f.write(" ")
			f.write("\n")

def load_abnormal_data(filename):
	"""
	加载故障对数据
	"""
	abnormal_data = []
	with open(filename, "r") as f:
		lines = f.readlines()
		for line in lines:
			line = tuple([np.float64(num) for num in line.strip().split(" ")])
			abnormal_data.append(line)

	return abnormal_data

def save_GAresult(filename, GAresult):
	with open(filename, "a+", encoding="utf-8") as f:
		for i in GAresult[2]:
			f.write(str(i) + " ")
		f.write(str(GAresult[0]) + "\n")
