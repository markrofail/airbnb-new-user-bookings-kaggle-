import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import decomposition
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

from src.helpers import paths
from src.models.train_model import load_data

####################################################
# Building Model #
####################################################
LE = LabelEncoder()


def train_gridSearch():

    X_train, Y_train, _, _ = load_data()
    Y_train = LE.fit_transform(Y_train)

    # Grid Search - Used to find best combination of parameters
    XGB_model = xgb.XGBClassifier(objective='multi:softprob',
                                  subsample=0.5, colsample_bytree=0.5, seed=0)
    param_grid = {'max_depth': [3, 4], 'learning_rate': [0.1, 0.3], 'n_estimators': [25, 50]}
    model = GridSearchCV(estimator=XGB_model, param_grid=param_grid,
                         scoring='accuracy', verbose=10, n_jobs=1, iid=True, refit=True, cv=3)

    model.fit(X_train.values, Y_train)

    print("Best score: %0.3f" % model.best_score_)
    print("Best parameters set:")

    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

####################################################
# Make predictions #
####################################################


def train():
    X_train, Y_train, _, _ = load_data()
    Y_train = LE.fit_transform(Y_train)

    model = xgb.XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=25, verbosity=2)
    model.fit(X_train.values, Y_train)

    with open(paths.models.forest(), 'wb') as file:
        pickle.dump(model, file)


def predict():
    X_train, Y_train, X_test, Y_test, id_test = load_data(test_ids=True)
    Y_test = pd.DataFrame(data=['NDF' for x in range(len(id_test))])

    LE.fit(Y_train)
    Y_test = LE.transform(Y_test)

    with open(paths.models.forest(), 'rb') as file:
        model = pickle.load(file)

    # Make predictions
    y_pred = model.predict_proba(X_test.values)

    # Taking the 5 classes with highest probabilities
    ids = []  # list of ids
    cts = []  # list of countries
    for i in range(len(id_test)):
        idx = id_test[i]
        ids += [idx] * 5
        cts += LE.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()

    # Generate submission
    print("Outputting final results...")
    sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
    sub.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    predict()
