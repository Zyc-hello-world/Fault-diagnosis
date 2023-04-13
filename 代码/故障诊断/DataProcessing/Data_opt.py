"""
对原始的数据进行操作
将四维数据转成2维
将四维数据转成[样本，width， channel]

划分训练集和测试集
"""
from scipy.signal import stft
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from keras.preprocessing import image
def axis4_axis3(data, is_onehot=True):
	"""将四维的原始数据转换成【样本数,宽度,通道】

	Args:
		data_4axis ([array]): 只含选择测点的原始数据[nodes, faults, mentos, features]
  
	Returns:
		[array]: [None, features, channel]
	"""
	faults = data.shape[1]
	mento = data.shape[2]
	features = data.shape[3]-1

	data = np.swapaxes(data, 0, 1)
	data = np.swapaxes(data, 1, 2)
	data = np.swapaxes(data, 2, 3)

	x_data = data.reshape([faults*mento, features+1, -1])
	y_data = x_data[:, -1][:, :1].reshape([-1, 1])
	x_data = x_data[:, :-1,:]

	if is_onehot:
		y_data = onehot_encode(y_data)

	return x_data, y_data

def axis4_axis_2(data_4axis_label, is_onehot=True):
	"""将四维数据转成二维，且每个样本包含了所有测点的信息

	Args:
		data_4axis ([array]): 原始数据[nodes, faults, mentos, features]
		stps_num ([list]): 测点名对应的测点的编号

	Returns:
		x_data[array]: 将所有同一测点下的故障进行展开维二维数组
		y_data[array]: 带标签的二维数据，未经过onehot编码
	"""
	x_data, y_data = axis4_axis3(data_4axis_label, False)
	x_data = np.swapaxes(x_data, 1, 2)
	x_data = x_data.reshape(x_data.shape[0], -1)
	if is_onehot:
		y_data = onehot_encode(y_data)
	return x_data, y_data

def onehot_encode(y_data):
	"""将标签的序号转为onehot编码

	Args:
		y_data ([array]): [None, 1]
	"""
	enc = OneHotEncoder()
	enc.fit(y_data)
	y_data_onehot = enc.transform(y_data).toarray()
	return y_data_onehot

def split_train_test(x_data, y_data, mentos, test_ratio=0.2):
	"""将数据划分为训练集和测试集

	Args:
		x_data ([type]): [description]可以是二维，也可以是四维的输入
		y_data ([type]): [description]
		test_ratio (float, optional): [description]. Defaults to 0.2.

	Returns:
		[type]: [description]
	"""
	if test_ratio == 0:
		return x_data, y_data, None, None
	np.random.seed()
	train_indices = []
	test_indices = []
	test_size = int(mentos * test_ratio)
	for start in range(0, len(x_data), mentos):
		number = range(start, start+mentos)
		shuffle_indices = np.random.permutation(number)
		test_indices.extend(shuffle_indices[:test_size])
		train_indices.extend(shuffle_indices[test_size:])

	x_train = x_data[train_indices]
	y_train = y_data[train_indices]

	x_test = x_data[test_indices]
	y_test = y_data[test_indices]
	return x_train, y_train, x_test, y_test

def get_batch_data(x_train, y_train, batch_size):
	
	batch_list = []
	batch_num = len(x_train) // batch_size + len(x_train) % batch_size
	for i in range(batch_num):
		start = i * batch_size
		end = min(start+batch_size, len(x_train))
		batch_x = x_train[start:end]
		batch_y = y_train[start:end]
		
		batch_list.append((batch_x, batch_y))
		
	return batch_list

def max_min_norm(x):
	"""输入为二维的数据，将其对每个测点按特征进行最大最小归一化

	Args:
		x ([array]): [node*faults*mento, features]
	"""
	if x is None:
		return x
	
	std = MinMaxScaler()
	x_norm = std.fit_transform(x)

	return x_norm
	


def shuffle_data(x, y):
	length = len(x)
	shuffle_indice = np.random.permutation(length)
	
	shuffle_x = x[shuffle_indice]
	shuffle_y = y[shuffle_indice]
	
	return shuffle_x, shuffle_y

def get_split_2axis_data(x2, y2, nodes, faults, mento, test_ratio=0.5):
	
	x_train,y_train, x_test, y_test = split_train_test(x2, y2, mento, test_ratio=test_ratio)
	print("this-------")

	return x_train, y_train, x_test, y_test

def get_split_3axis_data(x3, y3, nodes, faults, mento,test_ratio=0.5):
	print("三维数据形状")

	x_train, y_train , x_test, y_test = split_train_test(x3, y3, mento, test_ratio=test_ratio)

	return x_train, y_train , x_test, y_test


def read_img(path):
    
    
    data_gen = image.ImageDataGenerator(rescale=1./255)
    data = data_gen.flow_from_directory(directory=path, 
                                        target_size=(224, 224), 
                                        batch_size=64)
    
    return data


def get_image(raw_data, nodes_name2num_dict, file_folder, test_ratio=0.2):
    nodes, faults, mentos, features = raw_data.shape
    for node in range(nodes):
        for fault in range(faults):
            # folder_name = "./images/F{}".format(fault)
            # print(folder_name)
            # if not os.path.exists(folder_name):
            #     os.makedirs(folder_name)
            test_count = int(mentos * test_ratio)
            train_count = mentos - test_count
            train_or_test = "none"
            for i in range(mentos):
                x = raw_data[node][fault][i]
                # print(x.shape)
                f, t, nd = stft(x, fs=1.0, window=[1,1], nperseg=2)
                plt.pcolormesh(t, f, np.abs(nd))
                num = np.random.random()
                
                if num < (train_count / (train_count+test_count)):
                    # print(num, "train")
                    train_or_test = "/train/"
                    train_count -= 1
                else:
                    # print(num, "test")
                    train_or_test = "/test/"
                    test_count -= 1    
                folder_name = file_folder+train_or_test+"F{}".format(fault)
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)
                file_name = folder_name+"/node{}_{}.jpg".format(node, i)
                # plt.show()
                if not os.path.exists(file_name):
                    plt.savefig(file_name)
                
