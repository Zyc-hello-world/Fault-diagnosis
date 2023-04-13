
import GA_fitness
import save_data
import numpy as np

def all_comb(node_num):
	"""
	返回所有的测点组合
	"""
	res = [[]]
	for i in range(node_num):
		t = res.copy()
		length = len(res)
		for j in range(length):
			t = res[j].copy()
			t.append(i)
			res.append(t)
	return res

def get_best_res(node_num, fault_num, feat_alias_data):

	res = all_comb(node_num)
	result = {}
	for i, r in enumerate(res):
		fi_list, s, deg = GA_fitness.get_fitness(feat_alias_data, r, fault_num)
		result[i] = fi_list
	return result, res

