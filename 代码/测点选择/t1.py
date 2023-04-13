import save_data


feat_pair_data = CalculateAliasing.get_FaultPairData(feat_data, node_list)
pair_data = CalculateAliasing.get_FaultPairData(data, node_list)
def evaluate(d, theta):
	more_theta_ave = 0
	more_theta_num = 0

	less_theta_ave = 0
	less_theta_num = 0

	ave = 0
	ave_num = 0

	min_value = 1
	for node in d:
		for value in node:
			ave = ave + value
			ave_num = ave_num + 1
			min_value = min(min_value, value)
			if value > theta:
				more_theta_ave = more_theta_ave + value
				more_theta_num = more_theta_num + 1
			else:
				less_theta_ave = less_theta_ave + value
				less_theta_num = less_theta_num + 1
	more_theta_ave = more_theta_ave / more_theta_num
	less_theta_ave = less_theta_ave / less_theta_num
	ave = ave / ave_num

	return more_theta_ave, less_theta_ave, ave, min_value, len

file_alias_data = "alias_data.txt"
file_feat_alias_data = "feat_alias_data.txt"

#异常数据文件
file_feat_abnormal_data ="feat_abnormal_data.txt"
file_abnormal_data = "abnormal_data.txt"


feat_alias_data = save_data.load_alias_data(file_feat_alias_data)
feat_abnormal_data = save_data.load_abnormal_data(file_feat_abnormal_data)


alias_data = save_data.load_alias_data(file_alias_data)
abnormal_data = save_data.load_abnormal_data(file_abnormal_data)

feat = evaluate(feat_alias_data, 0.3)
no = evaluate(alias_data, 0.3)

print("经过特征提取")
print("大于0.3的故障对的个数:{}, 大于0.3的故障平均:{}, 小于0.3的故障平均:{}, 总平均:{}, 最小:{}".format(
	len(feat_abnormal_data), feat[0], feat[1], feat[2], feat[3]))

print("未经过特征提取")
print("大于0.3的故障对的个数:{}, 大于0.3的故障平均:{}, 小于0.3的故障平均:{}, 总平均:{}, 最小:{}".format(
	len(abnormal_data), no[0], no[1], no[2], no[3]))


