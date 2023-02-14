from multiprocessing import Process
from multiprocessing.pool import Pool

from data import get_data, create_headers, get_data_rnn
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


config.n_train_file = 5
config.n_test_file = 5
config.n_epochs = 1

headers = create_headers()

xy_train = np.array(get_data(config.n_train_file))
xy_test = np.array(get_data(m=config.n_test_file))

model = [
    DNN_Model([128, 64], ['relu', 'relu']),
    DNN_Model([256, 128], ['relu', 'relu']),
    DNN_Model([128, 64], ['elu', 'elu']),
    DNN_Model([256, 128], ['elu', 'elu'])
]

run_multi_process(model, xy_train, xy_test, headers)
exit()
del xy_test
del xy_train

xy_train = np.array(get_data_rnn(config.n_train_file))
xy_test = np.array(get_data_rnn(m=config.n_test_file))

model = [
    RNN_Model([128, 64], ['relu', 'relu']),
    RNN_Model([256, 128], ['relu', 'relu']),
    RNN_Model([128, 64], ['elu', 'elu']),
    RNN_Model([256, 128], ['elu', 'elu']),
    LSTM_Model([128, 64], ['relu', 'relu']),
    LSTM_Model([256, 128], ['relu', 'relu']),
    LSTM_Model([128, 64], ['elu', 'elu']),
    LSTM_Model([256, 128], ['elu', 'elu']),
]

for m in model:
    print(m.get_name())
    m.fit(xy_train, headers)
    m.test(xy_test, headers)

del xy_test
del xy_train

config.episode_duration = 20
xy_train = np.array(get_data_rnn(config.n_train_file))
xy_test = np.array(get_data_rnn(m=config.n_test_file))

model = [
    RNN_Model([128, 64], ['relu', 'relu']),
    RNN_Model([256, 128], ['relu', 'relu']),
    RNN_Model([128, 64], ['elu', 'elu']),
    RNN_Model([256, 128], ['elu', 'elu']),
    LSTM_Model([128, 64], ['relu', 'relu']),
    LSTM_Model([256, 128], ['relu', 'relu']),
    LSTM_Model([128, 64], ['elu', 'elu']),
    LSTM_Model([256, 128], ['elu', 'elu']),
]

for m in model:
    print(m.get_name())
    m.fit(xy_train, headers)
    m.test(xy_test, headers)
