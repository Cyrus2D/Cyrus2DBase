from data import get_data, create_headers, get_data_rnn
from models.config import config
from models.models import DNN_Model, RNN_Model, LSTM_Model

import numpy as np

config.n_train_file = 200
config.n_test_file = 100
config.n_epochs = 3

headers = create_headers()

xy_train = np.array(get_data(config.n_train_file))
xy_test = np.array(get_data(m=config.n_test_file))

model = [DNN_Model([128, 64], ['relu', 'relu']),
         DNN_Model([256, 128], ['relu', 'relu']),
         DNN_Model([128, 64], ['elu', 'elu']),
         DNN_Model([256, 128], ['elu', 'elu'])]

for m in model:
    m.fit(xy_train, headers)
    m.test(xy_test, headers)

del xy_test
del xy_train

xy_train = np.array(get_data_rnn(config.n_train_file))
xy_test = np.array(get_data_rnn(m=config.n_test_file))

model = [RNN_Model([128, 64], ['relu', 'relu']),
         RNN_Model([256, 128], ['relu', 'relu']),
         RNN_Model([128, 64], ['elu', 'elu']),
         RNN_Model([256, 128], ['elu', 'elu']),
         LSTM_Model([128, 64], ['relu', 'relu']),
         LSTM_Model([256, 128], ['relu', 'relu']),
         LSTM_Model([128, 64], ['elu', 'elu']),
         LSTM_Model([256, 128], ['elu', 'elu']), ]

for m in model:
    m.fit(xy_train, headers)
    m.test(xy_test, headers)
