import tensorflow as tf
from data import get_data, create_headers, Config, create_x_y_indexes, normalize_data
import numpy as np

TRAIN_PERCENT = 0.7
NX = 69


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
    model.add(tf.keras.layers.Dense(32, activation='relu', input_shape=(NX * episode_duration,)))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='linear'))

    model.compile(optimizer='adam', loss='mse')
    return model





headers = create_headers()
x_indexes, y_indexes = create_x_y_indexes(headers)
xy = np.array(get_data(300))

x = xy[:, x_indexes]
y = xy[:, y_indexes]

normalize_data(x, y)
xy = np.concatenate((x, y), axis=1)
np.random.shuffle(xy)

x = xy[:, :-2]
y = xy[:, -2:]

print(x.shape)
print(y.shape)

model = create_model_DNN(1)
model.fit(x, y, batch_size=64, epochs=1, validation_split=0.1)
model.save('model')
