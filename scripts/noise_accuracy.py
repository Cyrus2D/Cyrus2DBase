from math import floor

from data import create_headers, get_data
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle


def dist(x1, x2):
    return ((x1[:, 0] - x2[:, 0]) ** 2 + (x1[:, 1] - x2[:, 1]) ** 2) ** 0.5


def accuracy_plot():
    config = Config()
    headers = create_headers()
    xy = np.array(get_data())

    my_pos = (xy[:, headers["tm-9-full"]])[:, :-1]
    opp_pos_noise = (xy[:, headers["opp-5-noise"]])[:, :-1]
    opp_pos_full = (xy[:, headers["opp-5-full"]])[:, :-1]

    my_dist = dist(my_pos, opp_pos_full)
    pos_count = (xy[:, headers["opp-5-noise"]])[:, -1]

    max_pos_count = max(pos_count)
    max_dist = max(my_dist)

    pos_count_dist = [[0 for i in range(int(max_pos_count + 1))] for i in range(int(config.n_dist + 1))]
    counter = [[0 for i in range(int(max_pos_count + 1))] for i in range(int(config.n_dist + 1))]

    error = dist(opp_pos_noise, opp_pos_full)

    for i in range(xy.shape[0]):
        pc = int(pos_count[i])
        d = int((my_dist[i] / max_dist) * config.n_dist)
        e = error[i]

        pos_count_dist[d][pc] += e
        counter[d][pc] += 1

    pos_count_dist = np.array(pos_count_dist)
    counter = np.array(counter)

    for i in range(counter.shape[0]):
        for j in range(counter.shape[1]):
            counter[i][j] = 1 if counter[i][j] == 0 else counter[i][j]

    pos_count_dist = pos_count_dist / counter

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    ax.set_xlabel("dist")
    ax.set_ylabel("pos-count")
    ax.set_zlabel("error")

    X = np.arange(0, max_dist, max_dist / (config.n_dist + 1))
    Y = np.arange(0, max_pos_count + 1, 1.)
    X, Y = np.meshgrid(X, Y)
    Z = pos_count_dist.T

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, antialiased=False)

    pickle.dump(fig, open('figs/accuracy.pickle', 'wb'))
    plt.show()


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

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("N-Appearance")

    X = np.arange(-config.max_x, config.max_x, config.max_x * 2 / (config.n_x + 1))
    Y = np.arange(-config.max_y, config.max_y, config.max_y * 2 / (config.n_y + 1))
    X, Y = np.meshgrid(X, Y)
    Z = field_map.T
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, antialiased=False)

    pickle.dump(fig, open('figs/pos-5-full.pickle', 'wb'))
    plt.show()


accuracy_plot()
