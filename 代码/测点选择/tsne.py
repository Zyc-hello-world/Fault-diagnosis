
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE

def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 2.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig

def tsne(one_data, labels, title):
	"""
	one_data: 二维数据，样本*维度
	labels：样本的标签，用于绘图
	"""
	
	ts = TSNE(n_components=2, init='pca', random_state=0)
	# t-SNE降维
	reslut = ts.fit_transform(one_data)
	# 调用函数，绘制图像
	fig = plot_embedding(reslut, labels, 't-SNE Embedding of digits' + title)
	# 显示图像
	plt.show()

def plot_node_transient(data, node_num, fault_num):
	"""
	绘制node_num测点下，fault_num故障的曲线图
	"""
	t = np.arange(0, 0.001, 0.000002)
	
	x = np.array(data[node_num][fault_num]).copy()

	y = np.mean(x, 0)

	plt.plot(y,color='red')
	plt.title("第{}个测点下故障{}的时域图".format(node_num, fault_num))
	plt.show()
	