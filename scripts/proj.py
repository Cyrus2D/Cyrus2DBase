from data import create_headers, create_x_y_indexes, get_data, get_data_rnn, normalize_data_all, normalize_data_rnn_all
from models.models import DNN_Model, LSTM_Model
from models.config import config

import numpy as np

headers, _ = create_headers()

print('DNN')
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
    
    

for episode_duration in [5, 10, 15]:
    print(f'LSTM-{episode_duration}')
    config.episode_duration = episode_duration

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