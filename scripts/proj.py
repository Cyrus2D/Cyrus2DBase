import os
from multiprocessing import Pool

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from data import create_headers, create_x_y_indexes, get_data, get_data_rnn, normalize_data_all, normalize_data_rnn_all
from models.models import DNN_Model, LSTM_Model
from models.config import config
from matplotlib import cm

import numpy as np

err_range = []
for i in range(0, 200):
    err_range.append(i / 10)
# for i in range(5, 10):
#     err_range.append(i)
for i in range(20, 101, 10):
    err_range.append(i)


def train_and_test_models():
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


def dist(x1, x2):
    return ((x1[:, 0] - x2[:, 0]) ** 2 + (x1[:, 1] - x2[:, 1]) ** 2) ** 0.5


def make_last_seen_error():
    config.n_process = 20
    headers, _ = create_headers()
    xy = np.array(get_data(m=config.n_test_file))

    my_pos = (xy[:, headers["tm-9-full"]])[:, :2]
    opp_pos_noise = (xy[:, headers["opp-5-noise"]])[:, :2]
    opp_pos_full = (xy[:, headers["opp-5-full"]])[:, :2]

    error = dist(opp_pos_noise, opp_pos_full)
    my_dist = dist(my_pos, opp_pos_full)
    pos_count = (xy[:, headers["opp-5-noise"]])[:, 2]

    all = np.zeros((error.shape[0], 3))
    all[:, 0] = error
    all[:, 1] = my_dist
    all[:, 2] = pos_count

    np.savetxt(f"res/edp-data", all, delimiter=',')


def get_cmp(f1, f2):
    dic = {
        'r': [255, 0, 0],
        'g': [10, 145, 0],
        'b': [0, 0, 255],
        'y': [255, 0, 255]
    }

    N = 128
    vals1 = np.ones((N, 4))
    vals1[:, 0] = np.linspace(dic[f1][0] / 256, 1, N)
    vals1[:, 1] = np.linspace(dic[f1][1] / 256, 1, N)
    vals1[:, 2] = np.linspace(dic[f1][2] / 256, 1, N)

    vals2 = np.ones((N, 4))
    vals2[:, 0] = np.linspace(dic[f2][0] / 256, 1, N)
    vals2[:, 1] = np.linspace(dic[f2][1] / 256, 1, N)
    vals2[:, 2] = np.linspace(dic[f2][2] / 256, 1, N)

    vals = np.zeros((256, 4))
    vals[:128, :] = vals1
    vals[128:, :] = vals2[::-1, :]
    vals = vals[::-1, :]

    newcmp = ListedColormap(vals)

    return newcmp


def compare_heat_map(args):
    print(args[2], args[3])
    edp1 = args[0]
    edp2 = args[1]
    f1 = args[2]
    f2 = args[3]
    f1 = f1.split('/')[-1]
    f2 = f2.split('/')[-1]
    pos_count_dist_1 = [[0 for _ in range(int(30 + 1))] for _ in range(int(config.n_dist + 1))]
    counter_1 = [[0 for _ in range(int(30 + 1))] for _ in range(int(config.n_dist + 1))]

    max_dist = max(np.max(edp1[:, 1]), np.max(edp2[:, 1]))
    for i in range(edp1.shape[0]):
        pc = int(edp1[i][2])
        d = int((edp1[i][1] / max_dist) * config.n_dist)
        e = edp1[i][0]
        # e = edp1[i][0] / edp1[i][1]

        pos_count_dist_1[d][pc] += e
        counter_1[d][pc] += 1
    pos_count_dist_1 = np.array(pos_count_dist_1)
    counter_1 = np.array(counter_1)

    for i in range(counter_1.shape[0]):
        for j in range(counter_1.shape[1]):
            counter_1[i][j] = 1 if counter_1[i][j] == 0 else counter_1[i][j]

    pos_count_dist_2 = [[0 for _ in range(int(30 + 1))] for _ in range(int(config.n_dist + 1))]
    counter_2 = [[0 for _ in range(int(30 + 1))] for _ in range(int(config.n_dist + 1))]

    for i in range(edp2.shape[0]):
        pc = int(edp2[i][2])
        d = int((edp2[i][1] / max_dist) * config.n_dist)
        e = edp2[i][0]
        # e = edp2[i][0] / edp2[i][1]

        pos_count_dist_2[d][pc] += e
        counter_2[d][pc] += 1
    pos_count_dist_2 = np.array(pos_count_dist_2)
    counter_2 = np.array(counter_2)

    for i in range(counter_2.shape[0]):
        for j in range(counter_2.shape[1]):
            counter_2[i][j] = 1 if counter_2[i][j] == 0 else counter_2[i][j]

    pos_count_dist_2 = np.where((counter_1 < 3) * (counter_2 < 3), np.nan, pos_count_dist_2)
    pos_count_dist_1 = np.where((counter_1 < 3) * (counter_2 < 3), np.nan, pos_count_dist_1)

    pos_count_dist_1 /= counter_1
    pos_count_dist_2 /= counter_2

    Y = np.arange(0, max_dist, max_dist / (config.n_dist + 1))
    X = np.arange(0, 30 + 1, 1.)[:-5]
    X, Y = np.meshgrid(X, Y)
    Z = (pos_count_dist_2 - pos_count_dist_1)[:, :-5]

    min_z = np.nanmin(Z)
    max_z = np.nanmax(Z)
    v = max(max_z, abs(min_z))
    # Z = np.clip(Z, -1, 1)
    # c = []
    # for r in Z:
    #     cc = []
    #     for s in r:
    #         cc.append('r' if s > 0 else 'b')
    #     c.append(cc)

    fig, ax = plt.subplots()
    # print(Z)
    # print(max_z)
    # print(min_z)
    # print(v)
    # im = ax.imshow(Z, cmap='bwr')

    first_color = ''
    second_color = ''
    if f1.find('lstm') >= 0:
        first_color = 'r'
    elif f1.find('dnn') >= 0:
        first_color = 'g'
    else:
        first_color = 'b'

    if f2.find('lstm') >= 0:
        second_color = 'r'
    elif f2.find('dnn') >= 0:
        second_color = 'g'
    else:
        second_color = 'b'

    if first_color == second_color:
        second_color = 'y'

    cmap = get_cmp(first_color, second_color)

    ZB = np.where((counter_1 < 100) * (counter_2 < 100), 1, 0)
    im = ax.imshow(ZB, cmap='Greys', vmin=0, vmax=+1)
    # im = ax.imshow(Z, cmap='bwr'), vmin=-5, vmax=+5)
    im = ax.imshow(Z, cmap=cmap, vmin=-v, vmax=+v)
    ax.figure.colorbar(im, ax=ax, shrink=0.5)
    fig.tight_layout()
    ax.set_xlabel("pos-count")
    ax.set_ylabel("dist")

    # plt.show()
    plt.savefig(f"res/compare/E-{f1}-vs-E-{f2}")
    plt.close()
    # surf = ax.plot_surface(X, Y, Z, facecolors=c, antialiased=False)

    # pickle.dump(fig, open('figs/accuracy.pickle', 'wb'))
    # plt.show()


def make_heat_maps():
    file_list = os.listdir('res/')
    files = [
        f'res/{file}' for file in file_list if
        (file.startswith('edp') and file != 'edp-data')
    ]

    data = []
    for file in files:
        print(file)
        edp = np.genfromtxt(file, delimiter=',')
        data.append(edp)
    for i in range(len(data)):
        for j in range(len(data)):
            compare_heat_map((data[i], data[j], files[i], files[j]))


def pos_count_fig(data, files, pos_count):
    fig, ax = plt.subplots(1, 1)
    for edp, file in zip(data, files):
        counter = []
        for i in range(len(err_range) - 1):
            condition = (edp[:, 0] > err_range[i]) * (edp[:, 0] < err_range[i + 1]) * (edp[:, 2] == pos_count)
            counter.append(np.sum(np.where(condition, 1, 0)))
            # counter.append(np.sum(np.where(condition, 1, 0))
            #                + (np.sum(counter[-1] if i != 0 else 0)))

        counter.append(0)
        counter = np.array(counter)
        # print(counter / counter[-1])
        if file.find('lstm') != -1:
            color = 'red'
        elif file.find('rnn') != -1:
            color = 'blue'
        elif file.find('dnn') != -1:
            color = 'green'
        else:
            color = 'blue'
        ax.plot(err_range[:100], counter[:100] / np.sum(counter[:100]), c=color, label=file)
        # ax.plot(err_range, counter / np.sum(counter), color=color, label=file)

    ax.legend()
    plt.title(f"pc={pos_count}")
    plt.savefig(f'res/pc/{pos_count}.png')
    # plt.show()
    plt.close()


def draw_2d_based_on_poscount():
    file_list = os.listdir('res/')
    files = [
        f'res/{file}' for file in file_list if
        (file.startswith('edp') and file != 'edp-data')
    ]

    files = ['res/edp-data', 'res/edp-dnn-512-256-relu-relu-adam-mse-64',
             'res/edp-lstm-512-256-relu-relu-adam-mse-64-5']
    data = []
    for file in files:
        print(file)
        edp = np.genfromtxt(file, delimiter=',')
        data.append(edp)

    for i in range(20):
        pos_count_fig(data, files, i)


def compare3d(args):
    print(args[2], args[3])
    edp1 = args[0]
    edp2 = args[1]
    f1 = args[2]
    f2 = args[3]
    f1 = f1.split('/')[-1]
    f2 = f2.split('/')[-1]
    pos_count_dist_1 = [[0 for _ in range(int(30 + 1))] for _ in range(int(config.n_dist + 1))]
    counter_1 = [[0 for _ in range(int(30 + 1))] for _ in range(int(config.n_dist + 1))]

    max_dist = max(np.max(edp1[:, 1]), np.max(edp2[:, 1]))
    for i in range(edp1.shape[0]):
        pc = int(edp1[i][2])
        d = int((edp1[i][1] / max_dist) * config.n_dist)
        e = edp1[i][0]
        # e = edp1[i][0] / edp1[i][1]

        pos_count_dist_1[d][pc] += e
        counter_1[d][pc] += 1
    pos_count_dist_1 = np.array(pos_count_dist_1)
    counter_1 = np.array(counter_1)

    for i in range(counter_1.shape[0]):
        for j in range(counter_1.shape[1]):
            counter_1[i][j] = 1 if counter_1[i][j] == 0 else counter_1[i][j]

    pos_count_dist_2 = [[0 for _ in range(int(30 + 1))] for _ in range(int(config.n_dist + 1))]
    counter_2 = [[0 for _ in range(int(30 + 1))] for _ in range(int(config.n_dist + 1))]

    for i in range(edp2.shape[0]):
        pc = int(edp2[i][2])
        d = int((edp2[i][1] / max_dist) * config.n_dist)
        e = edp2[i][0]
        # e = edp2[i][0] / edp2[i][1]

        pos_count_dist_2[d][pc] += e
        counter_2[d][pc] += 1
    pos_count_dist_2 = np.array(pos_count_dist_2)
    counter_2 = np.array(counter_2)

    for i in range(counter_2.shape[0]):
        for j in range(counter_2.shape[1]):
            counter_2[i][j] = 1 if counter_2[i][j] == 0 else counter_2[i][j]

    pos_count_dist_2 = np.where((counter_1 < 3) * (counter_2 < 3), np.nan, pos_count_dist_2)
    pos_count_dist_1 = np.where((counter_1 < 3) * (counter_2 < 3), np.nan, pos_count_dist_1)

    pos_count_dist_1 /= counter_1
    pos_count_dist_2 /= counter_2

    Y = np.arange(0, max_dist, max_dist / (config.n_dist + 1))
    X = np.arange(0, 30 + 1, 1.)[:-5]
    X, Y = np.meshgrid(X, Y)
    Z = (pos_count_dist_2 - pos_count_dist_1)[:, :-5]

    min_z = np.nanmin(Z)
    max_z = np.nanmax(Z)
    v = max(max_z, abs(min_z))
    # Z = np.clip(Z, -1, 1)
    # c = []
    # for r in Z:
    #     cc = []
    #     for s in r:
    #         cc.append('r' if s > 0 else 'b')
    #     c.append(cc)

    ax = plt.gca(projection='3d')
    # print(Z)
    # print(max_z)
    # print(min_z)
    # print(v)
    # im = ax.imshow(Z, cmap='bwr')

    # first_color = ''
    # second_color = ''
    # if f1.find('lstm') >= 0:
    #     first_color = 'r'
    # elif f1.find('dnn') >= 0:
    #     first_color = 'g'
    # else:
    #     first_color = 'b'
    #
    # if f2.find('lstm') >= 0:
    #     second_color = 'r'
    # elif f2.find('dnn') >= 0:
    #     second_color = 'g'
    # else:
    #     second_color = 'b'
    #
    # if first_color == second_color:
    #     second_color = 'y'
    #
    # cmap = get_cmp(first_color, second_color)

    # ZB = np.where((counter_1 < 100) * (counter_2 < 100), 1, 0)
    # im = ax.imshow(ZB, cmap='Greys', vmin=0, vmax=+1)
    # # im = ax.imshow(Z, cmap='bwr'), vmin=-5, vmax=+5)
    # im = ax.imshow(Z, cmap=cmap, vmin=-v, vmax=+v)
    # ax.figure.colorbar(im, ax=ax, shrink=0.5)
    # fig.tight_layout()
    # ax.set_xlabel("pos-count")
    # ax.set_ylabel("dist")

    # plt.show()
    # plt.savefig(f"res/compare/E-{f1}-vs-E-{f2}")
    # plt.close()
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, antialiased=False)
    plt.savefig(f"res/compare3d/E-{f1}-vs-E-{f2}")
    plt.close()

    # pickle.dump(fig, open('figs/accuracy.pickle', 'wb'))
    # plt.show()


def make_3d():
    file_list = os.listdir('res/')
    files = [
        f'res/{file}' for file in file_list if
        (file.startswith('edp') and file != 'edp-data')
    ]

    data = []
    for file in files:
        print(file)
        edp = np.genfromtxt(file, delimiter=',')
        data.append(edp)
    for i in range(len(data)):
        for j in range(len(data)):
            compare3d((data[i], data[j], files[i], files[j]))

make_3d()
# draw_2d_based_on_poscount()
# make_last_seen_error()
# train_and_test_models()
# make_heat_maps()
