
def deg(sample_num, pre_labels):
	first = sample_num + pre_labels[:sample_num].sum() - pre_labels[sample_num:].sum()
	second = sample_num - pre_labels[:sample_num].sum() + pre_labels[sample_num:].sum()
	alias_degree = min(first, second) / 2 / sample_num
	return alias_degree

def evaluate(d, theta):
	"""
	d:alias_data
	"""
	more_theta_ave = 0
	more_theta_num = 0

	less_theta_ave = 0
	less_theta_num = 0

	ave = 0
	ave_num = 0

	min_value = 0
	for node in d:
		for value in node:
			ave = ave + value
			ave_num = ave_num + 1
			if value > theta:
				more_theta_ave = more_theta_ave + value
				more_theta_num = more_theta_num + 1
			else:
				if value == 0:
					min_value = min_value + 1
				less_theta_ave = less_theta_ave + value
				less_theta_num = less_theta_num + 1
	more_theta_ave = more_theta_ave / more_theta_num
	less_theta_ave = less_theta_ave / less_theta_num
	ave = ave / ave_num
	print("大于0.3的故障对的个数:{}, 大于0.3的故障平均:{}, 小于0.3的故障平均:{}, 总平均:{}, 0的数量:{}, 总数：{}".format(
	more_theta_num, more_theta_ave, less_theta_ave, ave, min_value, ave_num))
	return more_theta_ave, less_theta_ave, ave, min_value, ave_num