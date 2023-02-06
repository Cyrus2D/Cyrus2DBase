import tensorflow as tf
from data import get_data, create_headers, Config, create_x_y_indexes, normalize_data, episode_duration, \
    normalize_data_rnn
import numpy as np

TRAIN_PERCENT = 0.7
NX = 69


def create_model_RNN(episode_duration):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.SimpleRNN(32, activation='relu', input_shape=(episode_duration, NX)))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
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
    model.add(tf.keras.layers.Dense(32, activation='relu', input_shape=(NX * episode_duration,)))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='linear'))

    model.compile(optimizer='adam', loss='mse')
    return model


headers = create_headers()
x_indexes, y_indexes = create_x_y_indexes(headers)
xy = np.array(get_data(300))

print(xy.shape)

x = xy[:, :, x_indexes]
y = xy[:, -1, y_indexes]

print(x.shape)
print(y.shape)

normalize_data_rnn(x, y)

r_indexes = np.arange(x.shape[0])
np.random.shuffle(r_indexes)

x = x[r_indexes]
y = y[r_indexes]

print(x.shape)
print(y.shape)

model = create_model_RNN(episode_duration)
model.fit(x, y, batch_size=64, epochs=3, validation_split=0.1)
model.save('rnn-model')
