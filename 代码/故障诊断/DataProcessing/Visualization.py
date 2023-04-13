"""
可视化库
"""
from turtle import color
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from keras.utils import plot_model
import numpy as np
from numpy.core.fromnumeric import shape
from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns

def plot_model_results(acc, loss, val_acc, val_loss):
	"""可视化训练和验证集的损失函数和准确率

	Args:
		mod ([]): [keras model]
	"""
	# acc = history.history["class_acc"]
	# loss = history.history["loss"]
	
	# val_acc = history.history["val_class_acc"]
	# val_loss = history.history["val_loss"]
	epochs = range(1,len(acc)+1)
	
	plt.title("accuracy")
	plt.plot(epochs,acc,'r',label='Trainning acc')     #以epochs为横坐标，以训练集准确性为纵坐标
	plt.plot(epochs,val_acc,'b',label='Vaildation acc')
	plt.legend()

	plt.figure()   #创建一个新的图表
	plt.title("loss")
	plt.plot(epochs,loss,'r',label='Trainning loss')
	plt.plot(epochs,val_loss,'b',label='Vaildation loss')
	plt.legend()  ##绘制图例，即标明图中的线段代表何种含义
 
	plt.show()


def save_model_fig(model, save_file):
	"""将model的框架图保存

	Args:
		model ([type]): [description]
		save_file ([string]): [保存的文件路径]
	"""
	plot_model(model, to_file=save_file, show_shapes=True)


def vis_confusion_matrix(y_pred, y_test, faults_name, save_file):
	"""画出混淆矩阵

	Args:
		x_test ([type]): [description]
		y_test ([type]): [description]
		faults ([type]): [description]
	"""
	faults = len(faults_name)
	confusion_matrix = np.zeros([faults, faults])
		
	pred_loc = np.argmax(y_pred, 1).reshape(-1)

	y_class = np.argmax(y_test, 1).reshape(-1)


	for i in range(len(y_class)):
		confusion_matrix[y_class[i]][pred_loc[i]] += 1
	confusion_matrix = pd.DataFrame(confusion_matrix, columns=faults_name, index=faults_name)
	
	confusion_matrix.to_csv(save_file)
	return confusion_matrix


def vis_tsne(x, labels, fault_name):
	tsne = TSNE(n_components=3)
	y_label = np.argmax(labels, axis=1)
	low_dim_x = tsne.fit_transform(x)
	fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
	X, Y, Z = low_dim_x[:, 0], low_dim_x[:, 1], low_dim_x[:, 2]
	length = int(len(labels) / len(fault_name))
	count = len(fault_name)
	X_data = {}
	Y_data = {}
	for loc, seq in enumerate(y_label):
		if seq not in X_data:
			X_data[seq] = []
			Y_data[seq] = []
		X_data[seq].append(low_dim_x[loc])
		Y_data[seq].append(cm.rainbow(int(255/count*(y_label[loc]+1))))
	for (key, value) in X_data.items():
		value = np.array(value)
		ax.scatter(value[:,0], value[:,1],value[:,2], color=Y_data[key], label=fault_name[key])
	ax.set_xlabel("feature1")
	ax.set_ylabel("feature2")
	ax.set_zlabel("feature3")
	plt.legend(loc="upper right")
	plt.show()

