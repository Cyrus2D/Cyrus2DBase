from multiprocessing import Process


from data import  create_headers, get_data_rnn, create_x_y_indexes, normalize_data_rnn_all
from models.config import config

import numpy as np
import pytorch_sample as lstm


def create_labeled_y(xy, n_label, r, unum):
    headers, _ = create_headers()

    opp_pos_noise = np.array(xy[:, -1, headers[f'opp-{unum}-noise'][:2]])
    opp_pos_full = xy[:, -1, headers[f'opp-{unum}-full'][:2]]

    opp_pos_noise = np.where(opp_pos_noise == [-105, -105], [0, 0], opp_pos_noise)

    opp_err = opp_pos_noise - opp_pos_full
    opp_err = np.clip(opp_err, -r / 2, r / 2)
    index = np.floor((opp_err / r) * (n_label - 1)) + n_label / 2
    y_index = np.array(index[:, 0] * n_label + index[:, 1], dtype=np.uint32)

    y = np.zeros((opp_pos_noise.shape[0], n_label ** 2))
    y[np.arange(y_index.size), y_index] = 1

    return y


#
headers, _ = create_headers()
xy_train = np.array(get_data_rnn(config.n_train_file))

x_indexes, y_indexes = create_x_y_indexes(headers)

x = np.array(xy_train[:, :, x_indexes])
y = np.array(xy_train[:, -1, y_indexes])
y = create_labeled_y(xy_train, 20, 10, 5)

print(x.shape)
print(y.shape)

print('normalizing')
normalize_data_rnn_all(x)

r_indexes = np.arange(x.shape[0])
np.random.shuffle(r_indexes)

print('shuffling')
x = x[r_indexes]
y = y[r_indexes]

headers, _ = create_headers()
models = [
    lstm.TorchLSTM([512, 512], ['relu', 'relu']),
    # lstm.TorchLSTM([256, 128], ['relu', 'relu']),
    # lstm.TorchLSTM([512, 256], ['relu', 'relu']),

]
for model in models:
    model.train(x, y, headers)

