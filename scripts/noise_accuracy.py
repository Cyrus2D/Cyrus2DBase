from math import floor

from data import create_headers, get_data
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle


class Config:
    def __init__(self):
        self.n_x = 100
        self.n_y = 100

        self.max_x = 52.5
        self.max_y = 34.


def accuracy_plot():
    pass


def pos_plot():
    config = Config()
    headers = create_headers()

    xy = np.array(get_data())

    field_map = [[0 for i in range(config.n_y + 1)] for j in range(config.n_x + 1)]

    full_5_pos = xy[:, headers['opp-5-full']]

    out_of_field = 0
    for sample in full_5_pos:
        if abs(sample[0]) > config.max_x or abs(sample[1]) > 34:
            out_of_field += 1
            continue
        y = int((sample[1] + config.max_y) / (config.max_y * 2) * config.n_y)
        x = int((sample[0] + config.max_x) / (config.max_x * 2) * config.n_x)
        field_map[x][y] += 1
    field_map = np.array(field_map)
    print(out_of_field)
    print(max(field_map.flatten()))

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X = np.arange(-config.max_x, config.max_x, config.max_x * 2 / (config.n_x + 1))
    Y = np.arange(-config.max_y, config.max_y, config.max_y * 2 / (config.n_y + 1))
    X, Y = np.meshgrid(X, Y)
    Z = field_map.T
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, antialiased=False)

    pickle.dump(fig, open('figs/pos-5-full.pickle', 'wb'))
    plt.show()
