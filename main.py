import pandas as pd
import numpy as np
import scipy
import sklearn
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

dataframe = pd.read_csv('train.csv')
y = dataframe.iloc[:,1]
x = dataframe.iloc[:,2:]
x = np.array(x)
x = preprocessing.scale(x, axis=0)
#x = preprocessing.normalize(x, norm='l2', axis=1)
y = np.array(y)

lambdas = [0.01, 0.1, 1, 10, 100]

kf = KFold(n_splits=10)
kf.get_n_splits(x)

for lambdaNummer in lambdas:
	rmse = 0
	for train_index, test_index in kf.split(x):
		#print("TRAIN:", train_index, "TEST:", test_index)
		X_train, X_test = x[train_index], x[test_index]
		y_train, y_test = y[train_index], y[test_index]

		pred = Ridge(alpha=lambdaNummer)
		pred.fit(X_train,y_train)
		predY = pred.predict(X_test)
		rmse += np.sqrt(mean_squared_error(y_test, predY))
	print(str(rmse/10))