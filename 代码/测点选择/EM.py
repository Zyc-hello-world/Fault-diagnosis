import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np

from sklearn.mixture import GaussianMixture
from sklearn.datasets.samples_generator import make_blobs

#产生测试数据
def create_data(centers,num=100,std=0.7):
	X, labels_true = make_blobs(n_samples=num,centers=centers,cluster_std=std)
	return X, labels_true

#观察样本点分布
def plot_data(*data):
	X, labels_true = data
	labels = np.unique(labels_true)
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	colors = 'rgbyckm'
	for i, label in enumerate(labels):
		position = labels_true == label
		ax.scatter(X[position,0],X[position,1],label='cluster %d'%label,color=colors[i%len(colors)])
	ax.legend(loc='best',framealpha=0.5)
	ax.set_xlabel('x[0]')
	ax.set_ylabel('x[1]')
	ax.set_title('data')
	plt.show()
def GMM(X):
	
	# print(X.shape, type(X))
	
	
	gmm = GaussianMixture(n_components=2).fit(X) #指定聚类中心个数为2
	labels = gmm.predict(X)
	return labels
	# plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
	# plt.show()
	