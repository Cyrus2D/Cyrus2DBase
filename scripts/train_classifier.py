from multiprocessing import Process
from multiprocessing.pool import Pool
from time import sleep

from data import create_labeled_y, get_data, create_headers, get_data_rnn, create_x_y_indexes, normalize_data_rnn_all, normalize_data_all
from models.config import config
from models.models import DNN_Classifier_Model, DNN_Model, RNN_Model, LSTM_Model

import numpy as np



headers, _ = create_headers()

xy_train = np.array(get_data(config.n_train_file))
# xy_test = np.array(get_data(m=10))

x_indexes, y_indexes = create_x_y_indexes(headers)

x = np.array(xy_train[:, x_indexes])
y = create_labeled_y(xy_train, 20, 10, 5)

print(x.shape)
print(y.shape)

print('normalizing')
normalize_data_all(x)

r_indexes = np.arange(x.shape[0])
np.random.shuffle(r_indexes)

print('shuffling')
x = x[r_indexes]
y = y[r_indexes]

# xt = np.array(xy_test[:, x_indexes])
# normalize_data_all(xt)

model = [
    DNN_Classifier_Model([512, 256, 128, 64, 32],
                         ['relu', 'relu', 'relu', 'relu', 'relu'],
                         400),
    DNN_Classifier_Model([1024, 512],
                         ['relu', 'relu', 'relu', 'relu', 'relu'],
                         400),
]

for m in model:
    print(m.get_name())
    m.fit(x, y, headers)
    # m.test(xy_test, xt, headers)

