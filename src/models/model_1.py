import tensorflow as tf


class Model():
    def __init__(self, n_features):
        mlp = tf.keras.Sequential()
        mlp.add(tf.keras.layers.Input(shape=(n_features, )))
        mlp.add(tf.keras.layers.Dense(2048, activation='relu'))
        # mlp.add(tf.keras.layers.Dense(2048, activation='relu'))
        mlp.add(tf.keras.layers.Dropout(0.5))
        mlp.add(tf.keras.layers.Dense(1024, activation='relu'))
        mlp.add(tf.keras.layers.Dropout(0.5))
        mlp.add(tf.keras.layers.Dense(512, activation='relu'))
        mlp.add(tf.keras.layers.Dropout(0.5))
        mlp.add(tf.keras.layers.Dense(12, activation='softmax'))
        self.network = mlp
