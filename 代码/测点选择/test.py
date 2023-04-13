import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)
import DataProcess
import FeatureExtraction
import CalculateAliasing
import GeneticAlgorithm
import GA_fitness
import save_data
from tsne import tsne
import km
import EM
import cal_deg
import MyKmeans
import search_all
import Visualize


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
import os
import time
import pandas as pd

import geatpy as ea  # import geatpy
from MyProblem import MyProblem  # 导入自定义问题接口

"""

	data: 					从文件中获取到的数据，（测点数*故障类型*样本数*采样点数）type = array
	feat_data:				经过小波包特征提取的数据（测点*故障*样本*特征）type = array


	feat_pair_data:			经过特征提取的故障对数据，即每个测点下的所有可能组成的故障对（测点*故障对），
							每个故障对形如（i, j, pair_data），i,j为故障编号，pair_data为这两类故障的所有样本，i类在前
	pair_data:				未经过特征提取的故障对数据

	feat_abnormal_data:		经过特征提取的混叠度较大的异常数据，只存储节点号，故障编号和混叠度
	abnormal_data:			未经过特征提取的混叠度较大的异常数据，同上

	alias_matrix:			混叠表（测点*混叠度）type = list
	node_list:				节点的编号顺序
	fault_name:				故障的名字

"""

# 第一个电路的正则表达式为  ".*V\((\d+)\)@*", LDA=8
# 第二个电路的正则表达式为  ".*V\(n\d+\)@*"
# 后四个为				  "(.*)@.*"
reference_index = 6


data_info_dict = {"data_folder":["/input/data2/reference1/", "/input/KKCV-GA/caseII/", "/input/leapfrog_filter/hard_fault/",
								"/input/Sallen-Key/soft-fault50/", "/input/Four-opamp/soft-fault50/", 
								"/input/leapfrog_filter/soft-fault50/", "/input/ellipitic_filter/soft-fault50/"], 
				"nodes" : [11, 14, 12, 4, 8, 12, 14], 
				"faults" : [19, 45, 17, 13, 13, 24, 34], 
				"mento" : [30, 50, 50, 100, 200, 200, 200, 200], 
				"features": [50, 50, 50, 400, 600, 600, 500, 500],
				# 时间单位为s
				"simulate_time": [0.001, 0.001, 0.001, 0.00008, 0.0003, 0.001, 0.001],
				"circut_name": ["ref1", "case-II","hard", "Sallen-Key", "Four-opamp", "leapfrog_filter", "ellipitic_filter"]}
"""
data_info_dict = {"data_folder":["/input/leapfrog_filter/soft-fault50/", "/input/ellipitic_filter/soft-fault50/"], 
				"nodes" : [12, 14], 
				"faults" : [23, 34], 
				"mento" : [200, 200], 
				"features": [600, 500],
				# 时间单位为s
				"simulate_time": [0.001, 0.005]}
"""
data_folder = BASE_DIR + data_info_dict["data_folder"][reference_index]
nodes = data_info_dict["nodes"][reference_index]
faults = data_info_dict["faults"][reference_index]
mento = data_info_dict["mento"][reference_index]
# 采样点数
samples = data_info_dict["features"][reference_index]
simulate_time = data_info_dict["simulate_time"][reference_index]
circuit_name = data_info_dict["circut_name"][reference_index]

print("-----------------------------")
print(f"当前的被测电路为{circuit_name}")
print("------------------------------")
n = 5			#小波包的分解层数
theta = 0.3
class_theta = 0		#设置混叠度阈值

pop_size = 20
max_iter = 50

# class_method = km.pre
class_method = EM.GMM
# class_method = MyKmeans.get_labels
class_method_name = class_method.__name__
print("正在使用的聚类算法{}".format(class_method_name))

# 原始数据

raw_4data_label, node_list, fault_name = DataProcess.get_data_from_folder(data_folder, nodes, mento, samples, simulate_time)
raw_4data = raw_4data_label[:,:,:,:-1]


raw_2data_label = DataProcess.four_axis_to_two_axis(raw_4data_label, faults, nodes, mento)
raw_2data = raw_2data_label[:, :-1]
label = raw_2data_label[:, -1]
# 参考文献的5层小波包分解

# ref1_4data = FeatureExtraction.get_ref1_featdata(raw_4data, 4)
# ref1_2data = ref1_4data.reshape((nodes* faults* mento, -1))


# 我的特征提取,3层小波包提取能量，波动系数，...
myfeat_4data = FeatureExtraction.get_data(raw_4data, 3)
myfeat_2data = myfeat_4data.reshape((nodes*faults*mento, -1))

# print("myfeat_4data.shape={}, myfeat_2data.shape={}".format(myfeat_4data.shape, myfeat_2data.shape))

print(node_list)
print(fault_name)
# raw+LDA降维 n_com = 2
def raw_LDA(raw_2data, label):
	lda_2data_label = FeatureExtraction.MyLDA(raw_2data, label)
	lda_2data = lda_2data_label[:, :-1]

	lda_4data_label = lda_2data_label.reshape((nodes, faults, mento, -1))
	lda_4data = lda_4data_label[:, :, :, :-1]
	return lda_4data
# lda_4data = raw_LDA(raw_2data, label, 5)

# ref1+LDA n_com = 2
def ref1_LDA(ref1_2data, label):
	ref1_lda_2data_label = FeatureExtraction.MyLDA(ref1_2data, label, 5)
	ref1_lda_2data = ref1_lda_2data_label[:, :-1]

	ref1_lda_4data_label = ref1_lda_2data_label.reshape((nodes, faults, mento, -1))
	ref1_lda_4data = ref1_lda_4data_label[:, :, :, :-1]
	print("ref1_2data")
	print(ref1_2data[:100])

	print("ref1_lda_2data_label")
	print(ref1_lda_2data_label[:100])

	return ref1_lda_4data

def myfeat_LDA(myfeat_2data, label):
	myfeat_lda_2data_label = FeatureExtraction.MyLDA(myfeat_2data, label, 8)
	myfeat_lda_2data = myfeat_lda_2data_label[:, -1]

	myfeat_lda_4data_label = myfeat_lda_2data_label.reshape((nodes, faults, mento, -1))
	myfeat_lda_4data = myfeat_lda_4data_label[:, :, :, :-1]
	return myfeat_lda_4data
myfeat_lda_4data = myfeat_LDA(myfeat_2data, label)

file_GAresult = "GAresult.txt"
feat_pair_data = CalculateAliasing.get_FaultPairData(myfeat_lda_4data, node_list)




feat_alias_data, feat_abnormal_data = CalculateAliasing.get_AliasMatrix(feat_pair_data, mento, theta, class_method)
# from save_data import save_alias_dataToCsv
# save_alias_dataToCsv(feat_alias_data, faults)

import GA_fitness


def change2Binay(res, nodes):
	pop = []
	for one in res:
		number = list(np.zeros(nodes))
		for num in one:
			number[num-1] = 1
		pop.append(number)
	return pop
def combine(n, k):
	res = []
	tmp = []
	def dfs(start, level, tmp):
		if n-start + 1 < level : return
		if level == 0: res.append(tmp[::])
		for i in range(start, n+1):
			tmp.append(i)
			dfs(i+1, level-1, tmp)
			tmp.pop()
	dfs(1, k, tmp)
	pop = change2Binay(res, n)
	return pop

def help():
	Y = []
	for i in range(1, nodes+1):
		
		print(f"测点数量为{i}时的最优解为")
		index = []
		res = []
		pop = combine(nodes, i)
		
		FI, FI_list = GA_fitness.get_fitness_for_popularity(feat_alias_data, pop, faults)
		FI = FI.reshape(-1)
		max_index = np.argmax(FI)
		Y.append(FI[max_index])
		
		for i in range(len(FI)):
			if FI[i] == FI[max_index]:
				index.append(i)
				temp = GA_fitness.get_number(pop[i])
				res.append((temp, FI[i]))
		print(res)
	X = np.array(range(1, nodes+1))
	import seaborn as sns

	plt.rcParams['font.sans-serif'] = ['SimHei']  
# Matplotlib中设置字体-黑体，解决Matplotlib中文乱码问题
	plt.rcParams['axes.unicode_minus'] = False    
# 解决Matplotlib坐标轴负号'-'显示为方块的问题
	sns.set(font='SimHei')                        
	plt.plot(X, Y)
	plt.xlabel("测点数量")
	plt.ylabel("FID")
	plt.show()
help()		




"""
FI, res = search_all.get_best_res(node_num, fault_num, alias_data)

min_len = 10
for i in range(len(res)):
	if len(res[i]) == 1:
		print(res[i], FI[i])
	if len(FI[i]) == 16 and len(res[i]) == 2:
		print(res[i], FI[i])
"""

def eva(pop):
	FI_list, s, fault_pair = GA_fitness.get_fitness_for_one_pop(feat_alias_data,pop, faults)
	print(FI_list, s)
	for i in range(45):
		if i not in FI_list:
			print(fault_name[i])
	print(len(FI_list) + s)

	



def NSGA_II(feat_alias_data, nodes, faults):
	"""===============================实例化问题对象============================"""
	problem = MyProblem(feat_alias_data, nodes, faults)  # 生成问题对象
	"""==================================种群设置==============================="""
	Encoding = 'BG'  # 编码方式
	NIND = 50  # 种群规模
	Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # 创建区域描述器
	population = ea.Population(Encoding, Field, NIND)  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
	  
	"""================================算法参数设置============================="""
	myAlgorithm = ea.moea_NSGA2_templet(problem, population)  # 实例化一个算法模板对象
	myAlgorithm.mutOper.Pm = 0.2  # 修改变异算子的变异概率
	myAlgorithm.recOper.XOVR = 0.9  # 修改交叉算子的交叉概率
	myAlgorithm.MAXGEN = 20  # 最大进化代数
	myAlgorithm.logTras = 1  # 设置每多少代记录日志，若设置成0则表示不记录日志
	myAlgorithm.verbose = True  # 设置是否打印输出日志信息
	myAlgorithm.drawing = 1  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
	"""==========================调用算法模板进行种群进化=========================
	调用run执行算法模板，得到帕累托最优解集NDSet以及最后一代种群。NDSet是一个种群类Population的对象。
	NDSet.ObjV为最优解个体的目标函数值；NDSet.Phen为对应的决策变量值。
	详见Population.py中关于种群类的定义。
	"""
	[NDSet, population] = myAlgorithm.run()  # 执行算法模板，得到非支配种群以及最后一代种群
	NDSet.save()  # 把非支配种群的信息保存到文件中

	"""==================================输出结果=============================="""

	print('用时：%s 秒' % myAlgorithm.passTime)
	print('非支配个体数：%d 个' % NDSet.sizes) if NDSet.sizes != 0 else print('没有找到可行解！')
	print(NDSet.Phen)

	print("最优适应度为{}， 最优个体为{}".format(NDSet.ObjV, NDSet.Chrom))
	print("最后一代种群")
	print(population.ObjV)
	print(population.Chrom)

NSGA_II(feat_alias_data, nodes, faults)


