import numpy as np
import geatpy as ea
from GA_fitness import get_fitness_for_popularity

class MyProblem(ea.Problem):
	def __init__(self, alias_matrix, nodes, faults, M=2):
		self.alias_matrix = alias_matrix
		self.faults = faults
		name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
		Dim = 1  # 初始化Dim（决策变量维数）
		maxormins = [1, -1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
		varTypes = [1]  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
		lb = [0]  # 决策变量下界
		ub = [2**nodes]  # 决策变量上界
		lbin = [0]  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
		ubin = [0]  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
		# 调用父类构造方法完成实例化
		ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
	
	def aimFunc(self, pop):  # 目标函数
		Vars = pop.Phen  # 得到决策变量矩阵， 解码后的值
		BG = pop.Chrom	 # 得到染色体矩阵，0，1
		print("----------")
		# 测点的数量  二维数组
		f1 = np.sum(pop.Chrom, axis=1).reshape((-1, 1))
		# 隔离的数量，二维数组
		f2, FI_list = get_fitness_for_popularity(self.alias_matrix, BG, self.faults)
		
		#        # 利用罚函数法处理约束条件
		#        idx1 = np.where(x1 + x2 < 2)[0]
		#        idx2 = np.where(x1 + x2 > 6)[0]
		#        idx3 = np.where(x1 - x2 < -2)[0]
		#        idx4 = np.where(x1 - 3*x2 > 2)[0]
		#        idx5 = np.where(4 - (x3 - 3)**2 - x4 < 0)[0]
		#        idx6 = np.where((x5 - 3)**2 + x4 - 4 < 0)[0]
		#        exIdx = np.unique(np.hstack([idx1, idx2, idx3, idx4, idx5, idx6])) # 得到非可行解的下标
		#        f1[exIdx] = f1[exIdx] + np.max(f1) - np.min(f1)
		#        f2[exIdx] = f2[exIdx] + np.max(f2) - np.min(f2)
		# 利用可行性法则处理约束条件
		
		b = np.hstack([f1, f2])

		pop.ObjV = np.hstack([f1, f2])  # 把求得的目标函数值赋值给种群pop的ObjV