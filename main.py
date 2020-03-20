import pandas as pd
import numpy as np
import scipy
import sklearn
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import csv

dataframe = pd.read_csv('train.csv')
y = dataframe.iloc[:,1]
x = dataframe.iloc[:,2:]
x = np.array(x)
y = np.array(y)

lambdas = [0.01, 0.1, 1, 10, 100]

kf = KFold(n_splits=10)
kf.get_n_splits(x)

finalRMSE = []
for lambdaNummer in lambdas:
	rmse = []
	for train_index, test_index in kf.split(x):
		X_train, X_test = x[train_index], x[test_index]
		y_train, y_test = y[train_index], y[test_index]

		pred = Ridge(alpha=lambdaNummer, fit_intercept=False)
		pred.fit(X_train,y_train)
		predY = pred.predict(X_test)
		rmse.append(np.sqrt(mean_squared_error(y_test, predY)))
	print(str(np.mean(rmse)))
	finalRMSE.append(np.mean(rmse))

with open('./submission.csv', 'w', newline='') as csvfile:
	submission_file = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
	for entry in finalRMSE:
		submission_file.writerow([entry])
