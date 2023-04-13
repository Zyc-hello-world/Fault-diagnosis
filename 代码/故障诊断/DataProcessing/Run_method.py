from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
from DataProcessing import Data_opt
def kfold_test(x, y, fault_names, method, kfolds=5):
	kfolds = 5
	kf = KFold(n_splits=kfolds, shuffle=True, random_state=2022)
	test_acc = np.zeros(kfolds)
	for k, (train_ind, test_ind) in enumerate(kf.split(x)):
#for i in range(kfolds):
#x_train ,y_train , x_test, y_test = Data_opt.get_split_2axis_data(x2, y2, len(stps_num), faults, mento, test_ratio=0.3)
		x_train = x[train_ind]
		y_train = y[train_ind]
		x_test = x[test_ind]
		y_test = y[test_ind]
		test_acc[k] = method(x_train, y_train, x_test, y_test, fault_names)
		print("第{}次的准确率为{}".format(k+1, test_acc[k]))
	plt.plot(test_acc)
	plt.show()
	print(test_acc)
	print("{}次测试的平均准确率为{}".format(kfolds, test_acc.mean()))
