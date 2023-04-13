from sklearn import datasets

from sklearn.cluster import KMeans

# Loading dataset

def pre(data):

	model = KMeans(n_clusters=2)

# Fitting Model

	
	model.fit(data)# Predicitng a single input

	all_predictions = model.predict(data)

	return all_predictions

