import tensorflow as tf
from data import get_data
import numpy as np

def create_model_LSTM(episode_duration):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(256, input_shape=(episode_duration, 44)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='linear'))

    model.compile(optimizer='adam', loss='mse')
    return model

def create_model_DNN(episode_duration):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(44*episode_duration,)))
    model.add(tf.keras.layers.Dense(2, activation='linear'))

    model.compile(optimizer='adam', loss='mse')
    return model


x, y, episode_duration = get_data()
x = np.array(x)
print(x.shape)
# model = create_model_DNN(episode_duration)

# model.fit(x, y)
