
from tsne import tsne
import km
import MyKmeans
import DataProcess
import FeatureExtraction
import CalculateAliasing
import cal_deg
import numpy as np
import GA_fitness

import matplotlib.pyplot as plt
import IntegerCodeDict


y = [1,2,3,4,5]
plt.plot(y,color='red')
plt.show()

"""
data_folder = "E:/AllCode/Python/FaultData/data2/"
node_num = 11
fault_num = 33
sample_num = 30
n = 3


data, node_list, fault_name = DataProcess.get_data_from_file(data_folder, node_num, sample_num)

feat_data = FeatureExtraction.get_data(data, n)

feat_pair_data = CalculateAliasing.get_FaultPairData(feat_data, node_list)
pair_data = CalculateAliasing.get_FaultPairData(data, node_list)

def t(feat_pair_data):
	
	labels = [0] * 30 + [1] * 30
	d1 = feat_pair_data[0][0][2]

	l2= km.pre(d1)
	print(l2, cal_deg.deg(sample_num, l2))
	return (d1, list(l2))

res = []
for i in range(10):
	a = t(feat_pair_data)
	res.append(a)



from sklearn.cluster import KMeans
import numpy as np
data = np.random.rand(50, 3) #生成一个随机数据，样本大小为100, 特征数为3  
#假如我要构造一个聚类数为3的聚类器
a = []
for i in range(10): 
	estimator = KMeans(n_clusters=2)#构造聚类器 
	estimator.fit(data)#聚类 l
	abel_pred = estimator.labels_ #获取聚类标签 c
	entroids = estimator.cluster_centers_ #获取聚类中心 
	inertia = estimator.inertia_ # 获取聚类准则的总和
	a.append(abel_pred)
	print(abel_pred, np.sum(abel_pred))
"""