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
X_train, Y_train, X_test, Y_test = load_data()

le = LabelEncoder()
Y_train = le.fit_transform(Y_train)


# Training model
print("Training model...")

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
quit()
####################################################
# Make predictions #
####################################################
print("Making predictions...")

# Prepare test data for prediction
df_test.set_index('id', inplace=True)
df_test = pd.merge(df_test.loc[:, ['date_first_booking']], df_all,
                   how='left', left_index=True, right_index=True, sort=False)
X_test = df_test.drop('date_first_booking', axis=1, inplace=False)
X_test = X_test.fillna(-1)
id_test = df_test.index.values

# Make predictions
y_pred = model.predict_proba(X_test)

# Taking the 5 classes with highest probabilities
ids = []  # list of ids
cts = []  # list of countries
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 5
    cts += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()

# Generate submission
print("Outputting final results...")
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('./submission.csv', index=False)
