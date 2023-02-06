import tensorflow as tf
from data import get_data
import numpy as np

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


x, y, episode_duration = get_data(100)
x = np.array(x)
y = np.array(y)

print(x.shape)
print(y.shape)

np.random.shuffle(x)
np.random.shuffle(y)
