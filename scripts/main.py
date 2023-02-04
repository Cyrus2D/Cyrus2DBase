import tensorflow as tf
from data import get_data
import numpy as np

TRAIN_PERCENT = 0.7


def create_model_RNN(episode_duration):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(256, input_shape=(episode_duration, 44)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='linear'))

    model.compile(optimizer='adam', loss='mse')
    return model


def create_model_LSTM(episode_duration):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(256, input_shape=(episode_duration, 44)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='linear'))

    model.compile(optimizer='adam', loss='mse')
    return model


def create_model_DNN(episode_duration):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(256, activation='relu', input_shape=(44 * episode_duration,)))
    model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(44 * episode_duration,)))
    model.add(tf.keras.layers.Dense(2, activation='linear'))

    model.compile(optimizer='adam', loss='mse')
    return model


x, y, episode_duration = get_data()
x = np.array(x)
y = np.array(y)

np.random.shuffle(x)
np.random.shuffle(y)

print(x.shape)
print(y.shape)
model = create_model_RNN(episode_duration)

model.fit(x, y, batch_size=128, validation_split=0.3, epochs=10)
