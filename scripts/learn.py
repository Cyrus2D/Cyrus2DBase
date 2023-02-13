import tensorflow as tf
from data import get_data, create_headers, Config, create_x_y_indexes, normalize_data, episode_duration, \
    normalize_data_rnn, create_labeled_y, normalize_data_all, normalize_data_rnn_all, get_data_rnn
import numpy as np

TRAIN_PERCENT = 0.7
NX = 69
NY = 20


def create_model_RNN(episode_duration):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.SimpleRNN(32, activation='relu', input_shape=(episode_duration, NX)))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='linear'))

    model.compile(optimizer='adam', loss='mse')
    return model


def create_model_LSTM(episode_duration):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(128, activation='relu', input_shape=(episode_duration, NX)))
    # model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(2 * 11, activation='linear'))

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')
    return model


def create_model_DNN(episode_duration):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(2 ** 7, activation='relu', input_shape=(NX,)))
    model.add(tf.keras.layers.Dense(2 ** 6, activation='relu'))
    model.add(tf.keras.layers.Dense(2 * 11, activation='linear'))

    model.compile(optimizer='adam', loss='mse')
    return model


def create_model_DNN_softmax(episode_duration):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(2 ** 12, activation='relu', input_shape=(NX,)))
    model.add(tf.keras.layers.Dense(2 ** 10, activation='relu'))
    model.add(tf.keras.layers.Dense(NY ** 2, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# TODO SPLIT DATA
# TODO TEST AFTER LEARN
# TODO Error number of error in range
# TODO Error with different poscounts
# TODO Error with different distance
# TODO Different models (size, activition)
# TODO RUN EVERYTHING AUTOMATE
# TODO FILENAMES Discribe file


headers = create_headers()
x_indexes, y_indexes = create_x_y_indexes(headers)
print('arraying')
xy = np.array(get_data_rnn(50))

# y = create_labeled_y(xy, NY, 10)
print(xy.shape)

print('seperating')
# x = xy[:, :, x_indexes]
# y = xy[:, -1, y_indexes]
x = xy[:, x_indexes]
y = xy[:, y_indexes]

print(x.shape)
print(y.shape)

print('normalizing')
normalize_data_rnn_all(x, y)
# normalize_data(x, y)
# normalize_data(x)
# normalize_data_all(x, y)

r_indexes = np.arange(x.shape[0])
np.random.shuffle(r_indexes)

print('shuffling')
x = x[r_indexes]
y = y[r_indexes]

print(x.shape)
print(y.shape)

model = create_model_LSTM(episode_duration)
model.fit(x, y, batch_size=64, epochs=3, validation_split=0.1)
model.save('lstm-all-model')
