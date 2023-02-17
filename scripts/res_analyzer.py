import os
from multiprocessing.pool import Pool

import numpy as np
import matplotlib.pyplot as plt

from data import create_headers, get_data, dist
from models.config import config

err_range = []
for i in range(0, 200):
    err_range.append(i / 10)
# for i in range(5, 10):
#     err_range.append(i)
for i in range(20, 101, 10):
    err_range.append(i)


def data_error():
    config.n_process = 20
    headers, _ = create_headers()
    xy = np.array(get_data(m=200))

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
    return error


# data_error()
# exit()

file_list = os.listdir('res/')

files = [
    f'res/{file}' for file in file_list if
    (file.startswith('edp') and file != 'edp-data')
]

# files = [
#     f'res/{file}' for file in file_list if
#     (file.startswith('edp') and file.find('-elu') == -1 and file.find('dnn') != -1)
# ]

# files = [
#     f'res/{file}' for file in file_list if (file.startswith('edp')
#                                             and file.find('lstm') != -1)
# ]
# files = [
#     'res/edp-lstm-256-128-relu-relu-adam-mse-64',
# ]
files.append('res/edp-data')


def all_pos_counts(files):
    print(files)
    fig, ax = plt.subplots(1, 1)
    for file in files:
        edp = np.genfromtxt(file, delimiter=',')

        counter = []
        for i in range(len(err_range) - 1):
            # condition = (edp[:, 0] > err_range[i]) * (edp[:, 0] < err_range[i + 1])  # * (edp[:, 2] > 1)
            condition = (edp[:, 0] > err_range[i]) * (edp[:, 0] < err_range[i + 1]) * (edp[:, 2] >= 1)
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
            color = 'black'
        ax.plot(err_range, counter / np.sum(counter), label=file)
        # ax.plot(err_range, counter / np.sum(counter), color=color, label=file)

    ax.legend()
    plt.show()


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
            color = 'black'
        ax.plot(err_range[:100], counter[:100] / np.sum(counter), label=file)
        # ax.plot(err_range, counter / np.sum(counter), color=color, label=file)

    ax.legend()
    plt.title(f"pc={pos_count}")
    plt.savefig(f'res/pc/{pos_count}.png')
    # plt.show()
    plt.close()


def dist_fig(data, min_dist, max_dist):
    fig, ax = plt.subplots(1, 1)
    for edp in data:
        counter = []
        for i in range(len(err_range) - 1):
            condition = (edp[:, 0] > err_range[i]) * (edp[:, 0] < err_range[i + 1]) * (edp[:, 1] > min_dist) * (
                    edp[:, 1] < max_dist)
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
            color = 'black'
        ax.plot(err_range[:100], counter[:100] / np.sum(counter), label=file)
        # ax.plot(err_range, counter / np.sum(counter), color=color, label=file)

    ax.legend()
    plt.title(f"d=[{min_dist}, {max_dist}]")
    plt.savefig(f'res/d/{min_dist}.png')
    plt.close()


def pos_count_dist_fig(all):
    data = all[0]
    min_dist = all[1]
    max_dist = all[2]
    pos_count = all[3]

    print(pos_count, min_dist)
    fig, ax = plt.subplots(1, 1)
    for edp in data:
        counter = []
        for i in range(len(err_range) - 1):
            condition = (edp[:, 0] > err_range[i]) * (edp[:, 0] < err_range[i + 1]) * (edp[:, 1] > min_dist) \
                        * (edp[:, 1] < max_dist) * (edp[:, 2] == pos_count)
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
            color = 'black'
        ax.plot(err_range[:100], counter[:100] / np.sum(counter), label=file)
        # ax.plot(err_range, counter / np.sum(counter), color=color, label=file)

    ax.legend()
    plt.title(f"pc={pos_count}, d=[{min_dist}, {max_dist}]")
    plt.savefig(f'res/pcd/{pos_count}-{min_dist}.png')
    plt.close()


def compare_3d(all):
    edp1 = all[0]
    edp2 = all[1]
    f1 = all[2]
    f2 = all[3]
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

    pos_count_dist_2 = np.where((counter_1 < 100) * (counter_2 < 100), np.nan, pos_count_dist_2)
    pos_count_dist_1 = np.where((counter_1 < 100) * (counter_2 < 100), np.nan, pos_count_dist_1)

    pos_count_dist_1 /= counter_1
    pos_count_dist_2 /= counter_2

    print(max_dist)
    Y = np.arange(0, max_dist, max_dist / (config.n_dist + 1))
    X = np.arange(0, 30 + 1, 1.)[:-5]
    X, Y = np.meshgrid(X, Y)
    Z = (pos_count_dist_2 - pos_count_dist_1)[:, :-5]

    # Z = np.clip(Z, -1, 1)
    # c = []
    # for r in Z:
    #     cc = []
    #     for s in r:
    #         cc.append('r' if s > 0 else 'b')
    #     c.append(cc)

    fig, ax = plt.subplots()
    # im = ax.imshow(Z, cmap='bwr')
    ZB = np.where((counter_1 < 100) * (counter_2 < 100), 1, 0)
    im = ax.imshow(ZB, cmap='Greys', vmin=0, vmax=+1)
    # im = ax.imshow(Z, cmap='bwr'), vmin=-5, vmax=+5)
    im = ax.imshow(Z, cmap='bwr', vmin=-5, vmax=+5)
    ax.figure.colorbar(im, ax=ax, shrink=0.5)
    fig.tight_layout()
    ax.set_xlabel("pos-count")
    ax.set_ylabel("dist")

    # plt.show()
    plt.savefig(f"res/compare/E-{f1} - E-{f2}")
    plt.close()
    # surf = ax.plot_surface(X, Y, Z, facecolors=c, antialiased=False)

    # pickle.dump(fig, open('figs/accuracy.pickle', 'wb'))
    # plt.show()


# files = [
#     'res/edp-lstm-256-128-elu-elu-adam-mse-64',
#     'res/edp-data'
# ]
data = []
for file in files:
    print(file)
    edp = np.genfromtxt(file, delimiter=',')
    data.append(edp)

inp = []
for i in range(len(data)):
    for j in range(i, len(data)):
        print(i, j)
        inp.append((data[i], data[j], files[i], files[j]))

# all_pos_counts(files)
# for i in range(20):
#     print(i)
#     pos_count_fig(data, files, i)

# inp = []
# for i in range(0, 20):
#     for j in range(0, 20):
#         inp.append((data, j, j + 1, i))
#
pool = Pool(20)
pool.map(compare_3d, inp)
