import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)

from keras import layers, optimizers, losses
from keras import activations
from keras.engine.training import Model
from DataProcessing import Data_opt
from DataProcessing import Visualization, ReadData


def model_def(features, classes):
	input_tensor = layers.Input(shape=(features, 1))
	nb_lstm_outputs = 100  #神经元个数
	nb_time_steps = features  #时间序列长度
	nb_input_vector = 1 #每个特征向量化的长度
#l1 = layers.LSTM(units=nb_lstm_outputs, input_shape=(nb_time_steps, nb_input_vector))(input_tensor)
	l1 = layers.GRU(units=100, input_shape=(features, 1))(input_tensor)
	fc1 = layers.Dense(units=128, activation="relu")(l1)
	fc2 = layers.Dense(units=64, activation="relu")(fc1)
	out = layers.Dense(units=classes, activation="softmax")(fc2)
	model = Model(input_tensor, out, name="LSTM")
	return model

def train(x3, y3, classes, mento, circuit_name):
	save_file = "confusion_matrix/lstm_"+circuit_name+".csv"
	features = x3.shape[1]
	model = model_def(features, classes)
	model.compile(optimizer=optimizers.adam(0.001),
				loss=losses.categorical_crossentropy,
				metrics=["accuracy"])
	x_train, y_train, x_test, y_test = Data_opt.get_split_3axis_data(x3, y3, 1, classes, mento, test_ratio=0.3)
	history = model.fit(x_train, y_train, 
					 batch_size=128, epochs=500,
			validation_data=(x_test, y_test))
	acc = history.history["acc"]
	loss = history.history["loss"]
	val_acc = history.history["val_acc"]
	val_loss = history.history["val_loss"]
		
	Visualization.plot_model_results(acc, loss, val_acc, val_loss)
	test_acc = history.history["val_acc"][-1]

		# wpt_test = get_wide_data(x_test, n=5)[:,:,np.newaxis]
	y_pred = model.predict([x_test, x_test])
	print("测试准确率为{}".format(test_acc))
	confusion_matrix = Visualization.vis_confusion_matrix(y_pred, y_test, mydata.fault_names, save_file)
	print(confusion_matrix)
	return test_acc

if __name__ == "__main__":
	print("当前运行的模型是本文的模型")
	reference_index = 2
	is_norm = True

	data_info_dict = {"data_folder":["../input/Sallen-Key/soft-fault50", "../input/Four-opamp/soft-fault50", "../input/leapfrog_filter/soft-fault50/"], 
					"nodes" : [4, 8, 12], 
					"faults" : [13, 13, 23], 
					"mento" : [100, 200, 200], 
					"features": [400, 600, 600],
					# 时间单位为s
					"simulate_time": [0.00008, 0.0003, 0.001],
     				"circuit_name": ["Sallen-Key", "Four-opamp", "leapfrog_filter"],
         			"out_node": [["(V(4))"], ["V(8)"], ["V(12)"]]}

	data_folder = data_info_dict["data_folder"][reference_index]
	nodes = data_info_dict["nodes"][reference_index]
	faults = data_info_dict["faults"][reference_index]
	mento = data_info_dict["mento"][reference_index]
	# 采样点数
	features = data_info_dict["features"][reference_index]
	simulate_time = data_info_dict["simulate_time"][reference_index]



	circuit_name = data_info_dict["circuit_name"][reference_index]
	stps_name = data_info_dict["out_node"][reference_index]
	print("选择的测点为:{}".format(stps_name))
	save_file_name = "+".join(stps_name)

	channels = len(stps_name)

	# 原始数据
	mydata = ReadData.dataprocess(data_folder, nodes, faults, simulate_time, features, mento)
	mydata.get_data_from_folder()
	mydata.get_node_name()
	print("所有节点的名字为")
	print(mydata.nodes_name)

	print("所有故障的顺序为")
	print(mydata.fault_names)

	raw_data_with_label = mydata.raw_data[[mydata.nodes_name2num_dict[i] for i in stps_name]]
	raw_data = raw_data_with_label[:, :, :, :-1]
	stps_num = [mydata.nodes_name2num_dict[i] for i in stps_name]

	print(stps_num)

	print("源数据形状")
	print(raw_data_with_label.shape, raw_data.shape)

	
	# x2, y2 = axis4_axis_2(raw_data_with_label, is_onehot=True)
	x3, y3 = Data_opt.axis4_axis3(raw_data_with_label, is_onehot=True)
	if is_norm:
		x3_norm = x3.reshape((-1, features))
		x3_norm = Data_opt.max_min_norm(x3_norm)
	x3_norm = x3_norm.reshape((-1, features, len(stps_name)))

	print("三维数据格式为：{}-{}, {}".format(x3.shape, y3.shape, x3_norm.shape))
	

	test_acc = train(x3, y3, faults, mento, circuit_name)
	print("平均准确率为:{}".format(test_acc))
