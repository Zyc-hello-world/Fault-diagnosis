from CalculateAliasing import merge_AliasM
import numpy as np
import IntegerCodeDict  

def get_fitness_for_one_pop(alias_matrix, number, fault_num):
	"""
	计算种群的适应度函数
	alias_matrix:混叠度矩阵(测点*故障对对应的混叠度)
	number: 测点集合,二进制
	ICD:整数编码表（故障数*节点数）
	return :隔离的故障列表，总的混叠的故障对个数，总的混叠数
	"""
	FI_list = []
	s = 0
	degree = 0
	alias_fault_pair = []
	number = get_number(number)
	if len(number) != 0:
		cur_alias = (alias_matrix[number[0]]).copy()
		for alias in number:
			cur_alias = merge_AliasM(cur_alias, alias_matrix[alias])

		fault_table, degree, alias_fault_pair = IntegerCodeDict.get_fault_table(cur_alias, fault_num)
		FI_list = IntegerCodeDict.get_FIlist(fault_table)

	return FI_list, degree, alias_fault_pair

def get_fitness_for_popularity(alias_matrix, numbers, fault_num):
	"""Summary
	
	Args:
	    alias_matrix ([[],[],...,[]]): 所有测点构成的混叠表，每个测点用一个列表表示
	    numbers (array): 二进制编码的种群
	    fault_num (int): 故障数
	
	Returns:
	    TYPE: 该种群中每个个体的故障隔离数以及隔离故障名
	"""
	# return:
	FI = np.zeros((len(numbers), 1))
	FI_list = []
	for i, number in enumerate(numbers):
		fi_list, degree, alias_fault_pair = get_fitness_for_one_pop(alias_matrix, number, fault_num)
		FI[i][0] = len(fi_list) + 1 - degree
		FI_list.append(fi_list)

	return FI, np.array(FI_list)




def get_number(people):
	"""
	将二进制编码的个体转换为测点的数字集合,从0开始
	people: 二进制编码的个体
	return：测点集合
	"""
	#从左到右编号
	x = []
	for i in range(len(people)):
		if people[i] == 1:
			x.append(i)

	return x
