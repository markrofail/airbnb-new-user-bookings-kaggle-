import os

import numpy as np
import pandas as pd
import tensorflow as tf

from src.helpers import paths
from src.models.model_1 import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_data():
    X_train = pd.read_csv(paths.processed.train_dataset())
    Y_train = pd.read_csv(paths.processed.train_labels())
    X_train.drop(columns='id', inplace=True, errors='ignore')
    X_train.drop(columns='user_id', inplace=True, errors='ignore')
    Y_train.drop(columns='id', inplace=True, errors='ignore')
    Y_train.drop(columns='user_id', inplace=True, errors='ignore')

    X_test = pd.read_csv(paths.processed.test_dataset())
    Y_test = pd.read_csv(paths.processed.test_labels())
    X_test.drop(columns='id', inplace=True, errors='ignore')
    X_test.drop(columns='user_id', inplace=True, errors='ignore')
    Y_test.drop(columns='id', inplace=True, errors='ignore')
    Y_test.drop(columns='user_id', inplace=True, errors='ignore')

    return X_train, Y_train, X_test, Y_test


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = load_data()

    train_dt = tf.data.Dataset.from_tensor_slices((X_train.values, Y_train.values))
    train_dt = train_dt.batch(128)

    test_dt = tf.data.Dataset.from_tensor_slices((X_test.values, Y_test.values))
    test_dt = test_dt.batch(128)

    n_features = X_train.shape[1]
    model = Model(n_features).network

    model.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['categorical_accuracy'])
    model.fit(train_dt, epochs=5)
    model.evaluate(test_dt)
