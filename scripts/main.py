from multiprocessing import Process
from multiprocessing.pool import Pool
from time import sleep

from data import get_data, create_headers, get_data_rnn, create_x_y_indexes, normalize_data_rnn_all, normalize_data_all
from models.config import config
from models.models import DNN_Model, RNN_Model, LSTM_Model

import numpy as np


def combine_functions(f1, f2, f1_in, f2_in):
    f1(*f1_in)
    f2(*f2_in)


def run_multi_process(model, train, test, headers):
    train_functions = [m.fit for m in model]
    test_functions = [m.test for m in model]

    process = []
    for train, test in zip(train_functions, test_functions):
        p = Process(target=combine_functions, args=(train, test, (xy_train, headers), (xy_test, headers)))
        p.start()
        process.append(p)

    for p in process:
        p.join()


#
# config.n_train_file = 5
# config.n_test_file = 3
# config.n_epochs = 1
# config.n_process = 20

headers, _ = create_headers()

xy_train = np.array(get_data(config.n_train_file))
xy_test = np.array(get_data(m=config.n_test_file))

x_indexes, y_indexes = create_x_y_indexes(headers)

x = np.array(xy_train[:, x_indexes])
y = np.array(xy_train[:, y_indexes])

print(x.shape)
print(y.shape)

print('normalizing')
normalize_data_all(x, y)

r_indexes = np.arange(x.shape[0])
np.random.shuffle(r_indexes)

print('shuffling')
x = x[r_indexes]
y = y[r_indexes]

xt = np.array(xy_test[:, x_indexes])
normalize_data_all(xt)

model = [
    DNN_Model([128, 64], ['relu', 'relu']),
    DNN_Model([256, 128], ['relu', 'relu']),
    DNN_Model([512, 256], ['relu', 'relu']),
    DNN_Model([512, 256, 128, 64, 32], ['relu', 'relu', 'relu', 'relu', 'relu']),
    DNN_Model([256, 128, 64, 32], ['relu', 'relu', 'relu', 'relu']),
]

for m in model:
    print(m.get_name())
    m.fit(x, y, headers)
    m.test(xy_test, xt, headers)

del xy_test, xy_train, x, xt, y, r_indexes

config.episode_duration = 5
xy_train = np.array(get_data_rnn(config.n_train_file))
xy_test = np.array(get_data_rnn(m=config.n_test_file))

x_indexes, y_indexes = create_x_y_indexes(headers)

x = np.array(xy_train[:, :, x_indexes])
y = np.array(xy_train[:, -1, y_indexes])

print(x.shape)
print(y.shape)

print('normalizing')
normalize_data_rnn_all(x, y)

r_indexes = np.arange(x.shape[0])
np.random.shuffle(r_indexes)

print('shuffling')
x = x[r_indexes]
y = y[r_indexes]

xt = np.array(xy_test[:, :, x_indexes])
normalize_data_rnn_all(xt)

model = [
    LSTM_Model([256, 128], ['relu', 'relu']),
    LSTM_Model([512, 256], ['relu', 'relu']),
    LSTM_Model([128, 64, 32], ['relu', 'relu', 'relu']),
    LSTM_Model([512, 256, 128, 32], ['relu', 'relu', 'relu', 'relu']),
]
for m in model:
    print(m.get_name())
    m.fit(x, y, headers)
    m.test(xy_test, xt, headers)

del xy_test, xy_train, x, y, xt, r_indexes

config.episode_duration = 10
xy_train = np.array(get_data_rnn(config.n_train_file))
xy_test = np.array(get_data_rnn(m=config.n_test_file))

x_indexes, y_indexes = create_x_y_indexes(headers)

x = np.array(xy_train[:, :, x_indexes])
y = np.array(xy_train[:, -1, y_indexes])

print(x.shape)
print(y.shape)

print('normalizing')
normalize_data_rnn_all(x, y)

r_indexes = np.arange(x.shape[0])
np.random.shuffle(r_indexes)

print('shuffling')
x = x[r_indexes]
y = y[r_indexes]

xt = np.array(xy_test[:, :, x_indexes])
normalize_data_rnn_all(xt)

for m in model:
    print(m.get_name())
    m.fit(x, y, headers)
    m.test(xy_test, xt, headers)
del xy_test, xy_train, x, y, xt, r_indexes

config.episode_duration = 15
xy_train = np.array(get_data_rnn(config.n_train_file))
xy_test = np.array(get_data_rnn(m=config.n_test_file))

x_indexes, y_indexes = create_x_y_indexes(headers)

x = np.array(xy_train[:, :, x_indexes])
y = np.array(xy_train[:, -1, y_indexes])

print(x.shape)
print(y.shape)

print('normalizing')
normalize_data_rnn_all(x, y)

r_indexes = np.arange(x.shape[0])
np.random.shuffle(r_indexes)

print('shuffling')
x = x[r_indexes]
y = y[r_indexes]

xt = np.array(xy_test[:, :, x_indexes])
normalize_data_rnn_all(xt)

for m in model:
    print(m.get_name())
    m.fit(x, y, headers)
    m.test(xy_test, xt, headers)
del xy_test, xy_train, x, y, xt, r_indexes