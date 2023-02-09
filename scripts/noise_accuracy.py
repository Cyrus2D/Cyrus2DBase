'''
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
'''

from data import create_headers, get_data, Config, create_x_y_indexes, normalize_data, normalize_data_rnn, get_data_rnn
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
import tensorflow as tf


def dist(x1, x2):
    return ((x1[:, 0] - x2[:, 0]) ** 2 + (x1[:, 1] - x2[:, 1]) ** 2) ** 0.5


def dnn_vs_noise_accuracy():
    config = Config()
    xy = np.array(get_data(m=10))
    NX, NY, NZ, max_dist, max_pos_count = accuracy_plot(False, 100, xy)
    DX, DY, DZ, max_dist, max_pos_count = dnn_accuracy(False, 100, xy)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_xlabel("dist")
    ax.set_ylabel("pos-count")
    ax.set_zlabel("error")

    diff = NZ - DZ
    diff = np.clip(diff, -20, 20)

    surf = ax.plot_surface(NX, NY, diff, cmap=cm.coolwarm, antialiased=False)
    # surf = ax.plot_surface(NX, NY, NZ, color='r', antialiased=True)
    # surf = ax.plot_surface(NX, NY, DZ, color='b', antialiased=True)
    # pickle.dump(fig, open('figs/dnn-vs-poscount.pickle', 'wb'))

    plt.show()


def dnn_softmax_vs_noise_accuracy():
    config = Config()
    xy = np.array(get_data(m=100))
    NX, NY, NZ, max_dist, max_pos_count = accuracy_plot(False, 100, xy)
    DX, DY, DZ, max_dist, max_pos_count = dnn_softmax_accuracy(False, 100, xy)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_xlabel("dist")
    ax.set_ylabel("pos-count")
    ax.set_zlabel("error")

    diff = NZ - DZ
    diff = np.clip(diff, -20, 50)

    surf = ax.plot_surface(NX, NY, diff, cmap=cm.coolwarm, antialiased=False)
    # surf = ax.plot_surface(NX, NY, NZ, color='r', antialiased=True)
    # surf = ax.plot_surface(NX, NY, DZ, color='b', antialiased=True)
    # pickle.dump(fig, open('figs/dnn-vs-poscount.pickle', 'wb'))

    plt.show()


def dnn_softmax_vs_dnn_accuracy():
    config = Config()
    xy = np.array(get_data(m=100))
    NX, NY, NZ, max_dist, max_pos_count = dnn_accuracy(False, 100, xy)
    DX, DY, DZ, max_dist, max_pos_count = dnn_softmax_accuracy(False, 100, xy)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_xlabel("dist")
    ax.set_ylabel("pos-count")
    ax.set_zlabel("error")

    diff = NZ - DZ
    diff = np.clip(diff, -20, 50)

    # c = np.where(diff>0, [255,0,0], [0,0,255])
    c =[]
    for r in diff:
        cc = []
        for s in r:
            cc.append('r' if s > 0 else 'b')
        c.append(cc)
    print(c)

    surf = ax.plot_surface(NX, NY, diff, facecolors=c, antialiased=False)
    # surf = ax.plot_surface(NX, NY, NZ, color='r', antia   liased=True)
    # surf = ax.plot_surface(NX, NY, DZ, color='b', antialiased=True)
    # pickle.dump(fig, open('figs/dnn-vs-poscount.pickle', 'wb'))

    plt.show()


def rnn_vs_dnn_accuracy():
    config = Config()
    DX, DY, DZ, max_dist, max_pos_count = dnn_accuracy(False, 20)
    RX, RY, RZ, max_dist, max_pos_count = rnn_accuracy(False, 20)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    ax.set_xlabel("dist")
    ax.set_ylabel("pos-count")
    ax.set_zlabel("error")

    c =[]
    diff = DZ - RZ
    for r in diff:
        cc = []
        for s in r:
            cc.append('r' if s > 0 else 'b')
        c.append(cc)

    surf = ax.plot_surface(DX, DY, diff, facecolors=c, antialiased=False)
    # surf = ax.plot_surface(NX, NY, NZ, color='r', antialiased=True)
    # surf = ax.plot_surface(NX, NY, DZ, color='b', antialiased=True)
    # pickle.dump(fig, open('figs/dnn-vs-poscount.pickle', 'wb'))

    plt.show()


def rnn_accuracy(draw=True, n_data=100, xy=None):
    model = tf.keras.models.load_model('rnn-model')
    config = Config()
    headers = create_headers()
    if xy is None:
        xy = np.array(get_data_rnn(m=n_data))

    x_indexes, y_indexes = create_x_y_indexes(headers)
    x = np.array(xy[:, :, x_indexes])
    y = np.array(xy[:, -1, y_indexes])
    normalize_data_rnn(x)
    # print(x)

    my_pos = (xy[:, -1, headers["tm-9-full"]])[:, :-1]
    opp_pos_noise = model.predict(x)
    print(opp_pos_noise)
    opp_pos_noise[:, 0] *= config.max_x
    opp_pos_noise[:, 1] *= config.max_y
    print(opp_pos_noise)
    opp_pos_full = (xy[:, -1, headers["opp-5-full"]])[:, :-1]

    my_dist = dist(my_pos, opp_pos_full)
    pos_count = (xy[:, -1, headers["opp-5-noise"]])[:, -1]

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

    X = np.arange(0, max_dist, max_dist / (config.n_dist + 1))
    Y = np.arange(0, max_pos_count + 1, 1.)
    X, Y = np.meshgrid(X, Y)
    Z = pos_count_dist.T

    if draw:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        ax.set_xlabel("dist")
        ax.set_ylabel("pos-count")
        ax.set_zlabel("error")

        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, antialiased=False)

        pickle.dump(fig, open('figs/accuracy.pickle', 'wb'))
        plt.show()
    else:
        return X, Y, Z, max_dist, max_pos_count


def dnn_softmax_sample(draw=True, n_data=100, xy=None):
    model = tf.keras.models.load_model('softmax-model')
    config = Config()
    headers = create_headers()
    if xy is None:
        xy = np.array(get_data(m=n_data))

    x_indexes, _ = create_x_y_indexes(headers)
    x = np.array(xy[:, x_indexes])
    normalize_data(x)
    # print(x)
    # NN = 2578
    NN = 1239
    my_pos = (xy[:, headers["tm-9-full"]])[:, :-1]
    opp_pos_noise = model.predict(x)
    opp_pos_full = xy[NN, headers["opp-5-full"][:-1]]
    op = xy[NN, headers["opp-5-noise"][:-1]]
    sample = opp_pos_noise[NN]
    sample = sample.reshape(20, 20)
    print(opp_pos_full)
    print(op)
    X = np.arange(-5, +5, 10 / 20)
    Y = np.arange(-5, +5, 10 / 20)
    X, Y = np.meshgrid(X, Y)
    Z = sample
    if draw:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        ax.set_xlabel("dist")
        ax.set_ylabel("pos-count")
        ax.set_zlabel("error")

        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, antialiased=False)

        pickle.dump(fig, open('figs/accuracy.pickle', 'wb'))
        plt.show()
    else:
        return X, Y, Z, None, None


def dnn_softmax_accuracy(draw=True, n_data=100, xy=None):
    model = tf.keras.models.load_model('softmax-model')
    config = Config()
    headers = create_headers()
    if xy is None:
        xy = np.array(get_data(m=n_data))

    x_indexes, _ = create_x_y_indexes(headers)
    x = np.array(xy[:, x_indexes])
    normalize_data(x)
    # print(x)

    my_pos = (xy[:, headers["tm-9-full"]])[:, :-1]
    opp_pos_noise = model.predict(x)
    opp_pos_noise = opp_pos_noise.reshape(opp_pos_noise.shape[0], 20, 20)
    index = []
    for sample in opp_pos_noise:
        index.append(np.unravel_index(np.argmax(sample, axis=None), opp_pos_noise.shape[1:]))
    index = np.array(index)
    rel_pos = (index - 20 / 2) / (20 - 1) * 10
    opp_pos_noise = (xy[:, headers["opp-5-noise"]])[:, :-1] + rel_pos
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

    X = np.arange(0, max_dist, max_dist / (config.n_dist + 1))
    Y = np.arange(0, max_pos_count + 1, 1.)
    X, Y = np.meshgrid(X, Y)
    Z = pos_count_dist.T

    if draw:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        Z = np.clip(Z, -10, 50)

        ax.set_xlabel("dist")
        ax.set_ylabel("pos-count")
        ax.set_zlabel("error")

        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, antialiased=False)

        pickle.dump(fig, open('figs/accuracy.pickle', 'wb'))
        plt.show()
    else:
        return X, Y, Z, max_dist, max_pos_count


def dnn_accuracy(draw=True, n_data=100, xy=None):
    model = tf.keras.models.load_model('dnn-model')
    config = Config()
    headers = create_headers()
    if xy is None:
        xy = np.array(get_data(m=n_data))

    x_indexes, _ = create_x_y_indexes(headers)
    x = np.array(xy[:, x_indexes])
    normalize_data(x)
    # print(x)

    my_pos = (xy[:, headers["tm-9-full"]])[:, :-1]
    opp_pos_noise = model.predict(x)
    print(opp_pos_noise)
    opp_pos_noise[:, 0] *= config.max_x
    opp_pos_noise[:, 1] *= config.max_y
    print(opp_pos_noise)
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

    X = np.arange(0, max_dist, max_dist / (config.n_dist + 1))
    Y = np.arange(0, max_pos_count + 1, 1.)
    X, Y = np.meshgrid(X, Y)
    Z = pos_count_dist.T

    if draw:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        ax.set_xlabel("dist")
        ax.set_ylabel("pos-count")
        ax.set_zlabel("error")

        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, antialiased=False)

        pickle.dump(fig, open('figs/accuracy.pickle', 'wb'))
        plt.show()
    else:
        return X, Y, Z, max_dist, max_pos_count


def accuracy_plot(draw=True, n_data=100, xy=None):
    config = Config()
    headers = create_headers()
    if xy is None:
        xy = np.array(get_data(m=n_data))

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

    X = np.arange(0, max_dist, max_dist / (config.n_dist + 1))
    Y = np.arange(0, max_pos_count + 1, 1.)
    X, Y = np.meshgrid(X, Y)
    Z = pos_count_dist.T

    if draw:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        ax.set_xlabel("dist")
        ax.set_ylabel("pos-count")
        ax.set_zlabel("error")
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, antialiased=False)

        pickle.dump(fig, open('figs/accuracy.pickle', 'wb'))
        plt.show()
    else:
        return X, Y, Z, max_dist, max_pos_count


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


rnn_vs_dnn_accuracy()
