import numpy as np
import matplotlib.pyplot as plt

from data import create_headers, get_data, dist
from models.config import config

err_range = [
    0,
    0.1,
    0.3,
    0.4,
    0.6,
    0.8,
    0.9,
    1,
    2,
    3,
    4,
    5,
    10,
    15,
    20,
    30,
    40,
    50,
    100,
]

config.n_process = 20


def data_error():
    headers = create_headers()
    xy = np.array(get_data(m=100))

    opp_pos_noise = (xy[:, headers["opp-5-noise"]])[:, :-1]
    opp_pos_full = (xy[:, headers["opp-5-full"]])[:, :-1]

    error = dist(opp_pos_noise, opp_pos_full)
    return error


files = [
    'res/edp-dnn-128-64-relu-relu-adam-mse-64',
    'res/edp-rnn-128-64-relu-relu-adam-mse-64',
    'res/edp-lstm-128-64-relu-relu-adam-mse-64',
]
fig, ax = plt.subplots(1, 1)
for file in files:
    edp = np.genfromtxt(file, delimiter=',')

    counter = []
    for i in range(len(err_range) - 1):
        counter.append(np.sum(np.where((edp[:, 0] > err_range[i]) * (edp[:, 0] < err_range[i + 1]), 1, 0)))

    counter.append(0)
    counter = np.array(counter)
    print(counter / np.sum(counter))
    ax.plot(err_range, counter / np.sum(counter), label=file)

edp = data_error()
counter = []
for i in range(len(err_range) - 1):
    counter.append(np.sum(np.where((edp > err_range[i]) * (edp < err_range[i + 1]), 1, 0)))
counter.append(0)
counter = np.array(counter)
print(counter / np.sum(counter))
ax.plot(err_range, counter / np.sum(counter), label='data')

ax.legend()
plt.show()
