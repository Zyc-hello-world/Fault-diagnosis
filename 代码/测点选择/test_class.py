"""
测试kmeans聚类方法的相关指标
"""

import save_data
import CalculateAliasing
import km
import MyKmeans
import DataProcess
import os
import FeatureExtraction
import time

data_folder = "E:/AllCode/Python/FaultData/data2/"
node_num = 11
fault_num = 33
sample_num = 30
n = 3				#小波包的分解层数
theta = 0.3

data, node_list, fault_name = DataProcess.get_data_from_file(data_folder, node_num, sample_num)
feat_data = FeatureExtraction.get_data(data, n)

file_kmeans_feat_alias_data = "kmeans/feat_alias_data.txt"
file_kmeans_alias_data = "kmeans/alias_data.txt"

file_kmeans_feat_abnor_data = "kmeans/feat_abnor_data.txt"
file_kmeans_abnor_data = "kmeans/abnor_data.txt"

if not os.path.exists(file_kmeans_feat_alias_data):
	print("经过特征提取的文件数据保存")
	start = time.time()

	feat_pair_data = CalculateAliasing.get_FaultPairData(feat_data, node_list)
	feat_alias_data, feat_abnormal_data = CalculateAliasing.get_AliasMatrix(feat_pair_data, sample_num, theta, km.pre)
	

	save_data.save_alias_data(file_kmeans_feat_alias_data, feat_alias_data)
	save_data.save_abnormal_data(file_kmeans_feat_abnor_data, feat_abnormal_data)

	dis = time.time() - start
	m, s = divmod(dis, 60)
	h, m = divmod(m, 60)
	print("all time is {}h:{}m:{}s".format(h, m, s))
else:
	print("提取已保存的特征提取的数据")
	feat_alias_data = save_data.load_alias_data(file_kmeans_feat_alias_data)
	feat_abnormal_data = save_data.load_abnormal_data(file_kmeans_feat_abnor_data)

if not os.path.exists(file_kmeans_alias_data):
	print("未经过特征提取的文件数据保存")
	start = time.time()

	pair_data = CalculateAliasing.get_FaultPairData(data, node_list)
	alias_data, abnormal_data = CalculateAliasing.get_AliasMatrix(pair_data, sample_num, theta, km.pre)
	
	save_data.save_alias_data(file_kmeans_alias_data, alias_data)
	save_data.save_abnormal_data(file_kmeans_abnor_data, abnormal_data)

	dis = time.time() - start
	m, s = divmod(dis, 60)
	h, m = divmod(m, 60)
	print("all time is {}h:{}m:{}s".format(h, m, s))

else:
	print("提取以保存的未特征提取的数据")

	alias_data = save_data.load_alias_data(file_kmeans_alias_data)
	abnormal_data = save_data.load_abnormal_data(file_kmeans_abnor_data)






feat = evaluate(feat_alias_data, 0.3)
no = evaluate(alias_data, 0.3)

print("经过特征提取")


print("未经过特征提取")
print("大于0.3的故障对的个数:{}, 大于0.3的故障平均:{}, 小于0.3的故障平均:{}, 总平均:{}, 最小:{},总数:{}".format(
	len(abnormal_data), no[0], no[1], no[2], no[3], no[4]))


