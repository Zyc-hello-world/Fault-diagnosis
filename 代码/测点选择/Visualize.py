import pandas as pd
import os

def print_node_fault_table(node, alias_data, fault_num, node_list):
	table = [[0 for j in range(fault_num)] for i in range(fault_num)]
	k = 0
	for i in range(fault_num):
		for j in range(i+1, fault_num):
			table[i][j] = alias_data[node][k]
			k = k + 1
	columns = ["f"+str(i) for i in range(fault_num)]
	file = "fault_table/node" + str(node_list[node]) + ".csv"
	if not os.path.exists("fault_table"):
		os.makedirs("fault_table")
	
	df = pd.DataFrame(table , columns=columns)
	if not os.path.exists(file):
		df.to_csv(file)

	return df