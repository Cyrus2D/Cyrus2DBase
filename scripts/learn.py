import tensorflow as tf
from data import get_data, create_headers
import numpy as np

from noise_accuracy import Config

TRAIN_PERCENT = 0.7
NX = 46


def create_model_RNN(episode_duration):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.SimpleRNN(512, activation='relu', input_shape=(episode_duration, NX)))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='linear'))

    model.compile(optimizer='adam', loss='mse')
    return model


def create_model_LSTM(episode_duration):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(256, activation='relu', input_shape=(episode_duration, NX)))
    # model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='linear'))

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')
    return model


def create_model_DNN(episode_duration):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(512, activation='relu', input_shape=(NX * episode_duration,)))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='linear'))

    model.compile(optimizer='adam', loss='mse')
    return model


def create_x_y_indexes(headers: dict[str, list[int]]):
    x_indexes = []
    for key, value in headers.items():
        if key in ['cycle']:
            continue
        if key.find('full') != -1:
            continue
        x_indexes += value

    y_indexes = headers['opp-5-full'][:-1]
    return x_indexes, y_indexes

def normalize_data(x, y):
    config = Config()
    pos_x_i = [i for i in range(0, 69, 3)]
    pos_y_i = [i for i in range(1, 69, 3)]
    pos_count_i = [i for i in range(2, 69, 3)]

    x[:, pos_x_i] /= config.max_x
    x[:, pos_y_i] /= config.max_y
    x[:, pos_count_i] /= 30

    y[:, 0] /= config.max_x
    y[:, 1] /= config.max_y

headers = create_headers()
x_indexes, y_indexes = create_x_y_indexes(headers)
xy = np.array(get_data(10))

x = xy[:, x_indexes]
y = xy[:, y_indexes]

print(x)
print(y)
normalize_data(x, y)
print(x)
print(y)

print(x.shape)
print(y.shape)

np.random.shuffle(x)
np.random.shuffle(y)
