import tensorflow as tf
from models.config import config
from data import create_headers, create_x_y_indexes, normalize_data, normalize_data_all, dist, normalize_data_rnn_all
import numpy as np


class DNN_Model:
    def __init__(self, n_layers=[128, 64], activation=['relu', 'relu']):
        self.n_layers = n_layers
        self.activations = activation
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(n_layers[0], activation=activation[0], input_shape=(config.n_inputs,)))
        for l, a in zip(n_layers[1:], activation[1:]):
            model.add(tf.keras.layers.Dense(l, activation=a))
        model.add(tf.keras.layers.Dense(config.n_outputs, activation='linear'))
        model.compile(optimizer=config.optimizer, loss=config.loss)
        self.model = model

    def fit(self, x, y, headers):
        history = self.model.fit(x, y, batch_size=config.batch_size, epochs=config.n_epochs, validation_split=0.1)
        self.model.save(f"{self.get_name('model-')}")
        return history

    def test(self, xy, x, headers):
        my_pos = (xy[:, headers["tm-9-full"]])[:, :-1]
        opp_pos_noise = self.model.predict(x)[:, 8:10]
        opp_pos_noise[:, 0] *= config.max_x
        opp_pos_noise[:, 1] *= config.max_y
        opp_pos_full = (xy[:, headers["opp-5-full"]])[:, :-1]

        my_dist = dist(my_pos, opp_pos_full)
        pos_count = (xy[:, headers["opp-5-noise"]])[:, -1]
        error = dist(opp_pos_noise, opp_pos_full)

        all = np.zeros((error.shape[0], 3))
        all[:, 0] = error
        all[:, 1] = my_dist
        all[:, 2] = pos_count

        np.savetxt(f"{self.get_name('edp-')}", all, delimiter=',')

    def get_name(self, prefix=''):
        return f"{config.res_dir}/{prefix}dnn-{'-'.join(map(str, self.n_layers))}-{'-'.join(self.activations)}-{config.optimizer}-{config.loss}-{config.batch_size}"


class DNN_Classifier_Model:
    def __init__(self, n_layers=[128, 64], activation=['relu', 'relu'], n_out=400):
        self.n_layers = n_layers
        self.activations = activation
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(n_layers[0], activation=activation[0], input_shape=(config.n_inputs,)))
        for l, a in zip(n_layers[1:], activation[1:]):
            model.add(tf.keras.layers.Dense(l, activation=a))
        model.add(tf.keras.layers.Dense(n_out, activation='softmax'))
        model.compile(optimizer=config.optimizer, loss='categorical_crossentropy')
        self.model = model

    def fit(self, x, y, headers):
        history = self.model.fit(x, y, batch_size=config.batch_size, epochs=config.n_epochs, validation_split=0.1)
        self.model.save(f"{self.get_name('model-')}")
        return history

    def test(self, xy, x, headers):
        my_pos = (xy[:, headers["tm-9-full"]])[:, :-1]
        opp_pos_noise = self.model.predict(x)[:, 8:10]
        opp_pos_noise[:, 0] *= config.max_x
        opp_pos_noise[:, 1] *= config.max_y
        opp_pos_full = (xy[:, headers["opp-5-full"]])[:, :-1]

        my_dist = dist(my_pos, opp_pos_full)
        pos_count = (xy[:, headers["opp-5-noise"]])[:, -1]
        error = dist(opp_pos_noise, opp_pos_full)

        all = np.zeros((error.shape[0], 3))
        all[:, 0] = error
        all[:, 1] = my_dist
        all[:, 2] = pos_count

        np.savetxt(f"{self.get_name('edp-')}", all, delimiter=',')

    def get_name(self, prefix=''):
        return f"{config.res_dir}/{prefix}dnn-calssifier-{'-'.join(map(str, self.n_layers))}-{'-'.join(self.activations)}-{config.optimizer}-{config.loss}-{config.batch_size}"


class RNN_Model:
    def __init__(self, n_layers=[128, 64], activation=['relu', 'relu']):
        self.n_layers = n_layers
        self.activations = activation
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.SimpleRNN(n_layers[0], activation=activation[0],
                                            input_shape=(config.episode_duration, config.n_inputs)))
        for l, a in zip(n_layers[1:], activation[1:]):
            model.add(tf.keras.layers.Dense(l, activation=a))
        model.add(tf.keras.layers.Dense(config.n_outputs, activation='linear'))
        model.compile(optimizer=config.optimizer, loss=config.loss)
        self.model = model

    def fit(self, xy, headers):
        x_indexes, y_indexes = create_x_y_indexes(headers)

        x = xy[:, :, x_indexes]
        y = xy[:, -1, y_indexes]

        print(x.shape)
        print(y.shape)

        print('normalizing')
        normalize_data_rnn_all(x, y)

        r_indexes = np.arange(x.shape[0])
        np.random.shuffle(r_indexes)

        print('shuffling')
        x = x[r_indexes]
        y = y[r_indexes]

        history = self.model.fit(x, y, batch_size=config.batch_size, epochs=config.n_epochs)
        self.model.save(f"{self.get_name('model-')}")
        return history

    def test(self, xy, headers):
        x_indexes, _ = create_x_y_indexes(headers)
        x = np.array(xy[:, :, x_indexes])
        normalize_data_rnn_all(x)
        my_pos = (xy[:, -1, headers["tm-9-full"]])[:, :-1]
        opp_pos_noise = self.model.predict(x)[:, 8:10]
        opp_pos_noise[:, 0] *= config.max_x
        opp_pos_noise[:, 1] *= config.max_y
        opp_pos_full = (xy[:, -1, headers["opp-5-full"]])[:, :-1]

        my_dist = dist(my_pos, opp_pos_full)
        pos_count = (xy[:, -1, headers["opp-5-noise"]])[:, -1]
        error = dist(opp_pos_noise, opp_pos_full)

        all = np.zeros((error.shape[0], 3))
        all[:, 0] = error
        all[:, 1] = my_dist
        all[:, 2] = pos_count

        np.savetxt(f"{self.get_name('edp-')}", all, delimiter=',')

    def get_name(self, prefix=''):
        return f"{config.res_dir}/{prefix}rnn-{'-'.join(map(str, self.n_layers))}-{'-'.join(self.activations)}-{config.optimizer}-{config.loss}-{config.batch_size}-{config.episode_duration}"


class LSTM_Model:
    def __init__(self, n_layers=[128, 64], activation=['relu', 'relu']):
        self.n_layers = n_layers
        self.activations = activation
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(n_layers[0], activation=activation[0],
                                       input_shape=(config.episode_duration, config.n_inputs)))
        for l, a in zip(n_layers[1:], activation[1:]):
            model.add(tf.keras.layers.Dense(l, activation=a))
        model.add(tf.keras.layers.Dense(config.n_outputs, activation='linear'))
        model.compile(optimizer=config.optimizer, loss=config.loss)
        self.model = model

    def fit(self, x, y, headers):
        history = self.model.fit(x, y, batch_size=config.batch_size, epochs=config.n_epochs, validation_split=0.1)
        self.model.save(f"{self.get_name('model-')}")
        return history

    def test(self, xy, x, headers):
        my_pos = (xy[:, -1, headers["tm-9-full"]])[:, :-1]
        opp_pos_noise = self.model.predict(x)[:, 8:10]
        opp_pos_noise[:, 0] *= config.max_x
        opp_pos_noise[:, 1] *= config.max_y
        opp_pos_full = (xy[:, -1, headers["opp-5-full"]])[:, :-1]

        my_dist = dist(my_pos, opp_pos_full)
        pos_count = (xy[:, -1, headers["opp-5-noise"]])[:, -1]
        error = dist(opp_pos_noise, opp_pos_full)

        all = np.zeros((error.shape[0], 3))
        all[:, 0] = error
        all[:, 1] = my_dist
        all[:, 2] = pos_count

        np.savetxt(f"{self.get_name('edp-')}", all, delimiter=',')

    def get_name(self, prefix=''):
        return f"{config.res_dir}/{prefix}lstm-{'-'.join(map(str, self.n_layers))}-{'-'.join(self.activations)}-{config.optimizer}-{config.loss}-{config.batch_size}-{config.episode_duration}"
