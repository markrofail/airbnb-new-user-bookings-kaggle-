import os

import numpy as np
import pandas as pd
import tensorflow as tf

from src.helpers import paths
from src.models.model_1 import Model
import pickle

###############################################################################
# HYPER PARAMETERS
###############################################################################
config = paths.config.read(paths.config.config())
BETA1 = float(config['HYPERPARAMETERS']['BETA1'])
BETA2 = float(config['HYPERPARAMETERS']['BETA2'])
EPOCHS = int(float(config['HYPERPARAMETERS']['EPOCHS']))
EPSILON = float(config['HYPERPARAMETERS']['EPSILON'])
PATIENCE = config['HYPERPARAMETERS']['PATIENCE']
BASE_LINE = config['HYPERPARAMETERS']['BASE_LINE']
BATCH_SIZE = int(config['HYPERPARAMETERS']['BATCH_SIZE'])
LEARNING_RATE = float(config['HYPERPARAMETERS']['LEARNING_RATE'])


###############################################################################
# ENVIROMENTAL VARIABLES
###############################################################################
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
VERBOSE = config['ENVIROMENT_CONFIG']['VERBOSE']
SAVE_EVERY = int(config['ENVIROMENT_CONFIG']['SAVE_EVERY'])


def load_data(train_ids=False, test_ids=False):
    X_train = pd.read_csv(paths.processed.train_dataset())
    Y_train = pd.read_csv(paths.processed.train_labels())
    X_train.drop(columns='id', inplace=True, errors='ignore')
    X_train.drop(columns='user_id', inplace=True, errors='ignore')
    Y_train.drop(columns='id', inplace=True, errors='ignore')
    Y_train.drop(columns='user_id', inplace=True, errors='ignore')

    X_test = pd.read_csv(paths.processed.test_dataset())
    Y_test = pd.read_csv(paths.processed.test_labels())
    id_list = X_test.id.values
    X_test.drop(columns='id', inplace=True, errors='ignore')
    X_test.drop(columns='user_id', inplace=True, errors='ignore')
    Y_test.drop(columns='id', inplace=True, errors='ignore')
    Y_test.drop(columns='user_id', inplace=True, errors='ignore')

    if test_ids:
        return X_train, Y_train, X_test, Y_test, id_list
    return X_train, Y_train, X_test, Y_test


def format_predictions(predictions):
    Y_test = pd.read_csv(paths.processed.test_labels())
    user_ids = Y_test.pop('id')

    classes = list()
    for calss_x in list(Y_test.columns):
        classes.append(str(calss_x).replace("country_destination_", ""))

    result_df = pd.DataFrame(columns=['id', 'country'])
    for user_id, pred in zip(user_ids, predictions):
        classes_values = [classes[i] for i in np.argsort(pred)[-5:]]
        data = [(user_id, x) for x in classes_values]
        for x, y in data:
            result_df = result_df.append({'id':x, 'country':y}, ignore_index=True)
    print(result_df.head())
    return result_df


def run(epochs, batch_size=1, learning_rate=0.001, beta1=0.9, beta2=0.999,
        epsilon=1e-08, save_every=10, patience=5, baseline=2e-5, resume=False):
    print("Loading Data...")
    X_train, Y_train, _, _ = load_data()

    train_dt = tf.data.Dataset.from_tensor_slices((X_train.values, Y_train.values))
    train_dt = train_dt.shuffle(buffer_size=128, reshuffle_each_iteration=True)

    n_rows = X_train.shape[0]
    val_size = int(0.2*n_rows)

    n_batches = (n_rows-val_size)//batch_size
    n_batches_val = (val_size)//batch_size

    val_dt = train_dt.take(val_size)
    val_dt = val_dt.batch(batch_size)
    # val_dt = val_dt.repeat()

    train_dt = train_dt.skip(val_size)
    train_dt = train_dt.batch(batch_size)
    train_dt = train_dt.repeat()

    # Init callbacks
    cbs = list()
    # ModelCheckpoint callback: saves model every SAVE_EVERY
    checkpoint_path = paths.models._2048_1024_512()
    checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
    if checkpoint_path.exists() and not resume:
        paths.delete_folder(checkpoint_path)  # deletes file before training
    cbs.append(tf.keras.callbacks.ModelCheckpoint(
        str(checkpoint_path), save_freq='epoch'))

    cbs.append(tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                    patience=5, min_lr=0.001))

    # History callback: saves all losses
    cbs.append(tf.keras.callbacks.CSVLogger(
        str(checkpoint_path.with_suffix('.csv')), append=True, separator=','))

    print("Begin Training...")
    n_features = X_train.shape[1]
    model = Model(n_features).network

    print("# Compile")
    model.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['categorical_accuracy'])
    print("# Fit")
    model.fit(
        train_dt,
        validation_data=val_dt,
        steps_per_epoch=n_batches,
        epochs=epochs,
        callbacks=cbs,
        # validation_steps=n_batches_val
    )


def evaluate(batch_size=BATCH_SIZE):
    _, _, X_test, Y_test = load_data()

    test_dt = tf.data.Dataset.from_tensor_slices((X_test.values, Y_test.values))
    test_dt = test_dt.shuffle(buffer_size=128, reshuffle_each_iteration=True)
    test_dt = test_dt.batch(batch_size)

    n_features = X_test.shape[1]
    model = Model(n_features).network
    model.evaluate(test_dt)


def predict(batch_size=BATCH_SIZE):
    # _, _, X_test, Y_test = load_data()

    # test_dt = tf.data.Dataset.from_tensor_slices((X_test.values, Y_test.values))
    # test_dt = test_dt.shuffle(buffer_size=128, reshuffle_each_iteration=True)
    # test_dt = test_dt.batch(batch_size)

    # # n_features = X_test.shape[1]
    # # model = Model(n_features).network
    # path = str(paths.models._2048_1024_512(file=False))
    # model = tf.keras.models.load_model(path)

    # predictions = model.predict(test_dt, batch_size=None)

    # with open('filename.pickle', 'wb') as handle:
    #     pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('filename.pickle', 'rb') as handle:
        predictions = pickle.load(handle)
    result_df = format_predictions(predictions)
    result_df.to_csv(paths.results._2048_1024_512(), index=False)


if __name__ == "__main__":
    print('\n# Training for {} epoch(s)...'.format(EPOCHS))
    # run(
    #     beta1=BETA1,
    #     beta2=BETA2,
    #     epsilon=EPSILON,
    #     patience=PATIENCE,
    #     baseline=BASE_LINE,
    #     batch_size=BATCH_SIZE,
    #     save_every=SAVE_EVERY,
    #     learning_rate=LEARNING_RATE,
    #     epochs=EPOCHS,
    # )

    predict()
