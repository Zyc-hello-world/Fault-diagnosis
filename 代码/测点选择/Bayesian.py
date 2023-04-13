from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn import datasets

from data_package import Data2dAndLabels
import DataProcess
from data_package import all_data_preprocess, splite_data_preprocess


node_num = 11
sample_num = 30
data_folder = "E:/AllCode/Python/FaultData/data2/"

"""
data, node_list, fault_name = DataProcess.get_data_from_file(data_folder, node_num, sample_num)
X_train, y_train = Data2dAndLabels(data)
"""
# X_train = [[1,2,3], [4,5,6], [7,8,9]]


iris = datasets.load_iris()
X_train = iris.data
y_train = iris.target

method1 = preprocessing.StandardScaler().fit_transform
method2 = preprocessing.MinMaxScaler().fit_transform
method3 = preprocessing.MaxAbsScaler().fit_transform
method4 = preprocessing.Normalizer().fit_transform

# a = preprocessing.StandardScaler().fit_transform(X_train)

def one():
	standard_scaler_data = all_data_preprocess(X_train, method1)

	min_max_scaler_data = all_data_preprocess(X_train,method2)

	max_abs_scaler_data = all_data_preprocess(X_train, method3)

	normalizer_data = all_data_preprocess(X_train, method4)

	d = [X_train, standard_scaler_data, min_max_scaler_data, max_abs_scaler_data, normalizer_data]
	return d
def two():
	standard_scaler_data = splite_data_preprocess(X_train, method1)

	min_max_scaler_data = splite_data_preprocess(X_train,method2)

	max_abs_scaler_data = splite_data_preprocess(X_train, method3)

	normalizer_data = splite_data_preprocess(X_train, method4)

	d = [X_train, standard_scaler_data, min_max_scaler_data, max_abs_scaler_data, normalizer_data]
	return d
e = one()

"""
a = preprocessing.StandardScaler().fit_transform(X_train)

b = preprocessing.MinMaxScaler().fit_transform(X_train)
c = preprocessing.MaxAbsScaler().fit_transform(X_train)
d = preprocessing.Normalizer().fit_transform(X_train)
"""
# print(node_list, fault_name)



for i in e:
	gnb = GaussianNB()
	y_pred = gnb.fit(i, y_train).predict(i)

	print("Number of mislabeled points out of a total {} points : {}".format(X_train.shape[0], (y_train != y_pred).sum()))


"""
c1_down = X_train[:30]
c1_up = X_train[30:60]
sum = 0
for i in range(len(c1_down)):
	for j in range(len(c1_up[i])):
		if abs(c1_down[i][j] - c1_up[i][j]) < 0.001:
			sum = sum + 1
			print(i, j, c1_down[i][j], c1_up[i][j])

print(sum)

"""
