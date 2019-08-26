
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

trainX = correct_train[predictor].values
trainY = correct_train.Survived.values

MLPClassifier(hidden_layer_sizes=(100, ), activation=’relu’, solver=’adam’, alpha=0.0001, batch_size=’auto’, learning_rate=’constant’, learning_rate_init=0.001, power_t=0.5, max_iter=200)

parameters = {'solver': ['lbfgs'], 'max_iter': [1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000 ], 'alpha': 10.0 ** -np.arange(1, 10), 'hidden_layer_sizes':np.arange(10, 15), 'random_state':[0,1,2,3,4,5,6,7,8,9]}
clf = GridSearchCV(MLPClassifier(), parameters, n_jobs=-1)

clf.fit(trainX, trainY)
print(clf.score(trainX, trainY))
print(clf.best_params_)

correct_test = correct_data(test)
testX = correct_test[predictor].values
result = clf.predict(testX)

test["Survived"] = result
result = test[["PassengerId", "Survived"]]

result.to_csv('titanic_RandomForestClassifier_FamilySize.csv', index=False)