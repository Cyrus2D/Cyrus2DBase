import tensorflow as tf
from data import get_data


def create_model(episode_duration):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(256, input_shape=(episode_duration, 44)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='linear'))

    model.compile(optimizer='adam', loss='mse')
    return model


x, y, episode_duration = get_data()
model = create_model(episode_duration)

model.fit(x, y)
