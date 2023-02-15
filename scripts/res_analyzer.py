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
    headers = create_headers()
    xy = np.array(get_data(m=100))

    my_pos = (xy[:, headers["tm-9-full"]])[:, :-1]
    opp_pos_noise = (xy[:, headers["opp-5-noise"]])[:, :-1]
    opp_pos_full = (xy[:, headers["opp-5-full"]])[:, :-1]

    error = dist(opp_pos_noise, opp_pos_full)
    my_dist = dist(my_pos, opp_pos_full)
    pos_count = (xy[:, headers["opp-5-noise"]])[:, -1]

    all = np.zeros((error.shape[0], 3))
    all[:, 0] = error
    all[:, 1] = my_dist
    all[:, 2] = pos_count

    np.savetxt(f"res/edp-data", all, delimiter=',')
    return error


# data_error()
# exit()

file_list = os.listdir('res/')

# files = [
#     f'res/{file}' for file in file_list if (file.startswith('edp'))
# ]
# files = [
#     f'res/{file}' for file in file_list if (file.startswith('edp')
#                                             and file.find('lstm') != -1)
# ]
files = [
    'res/edp-lstm-256-128-relu-relu-adam-mse-64',
]
files.append('res/edp-data')


def all_pos_counts(files):
    fig, ax = plt.subplots(1, 1)
    for file in files:
        edp = np.genfromtxt(file, delimiter=',')

        counter = []
        for i in range(len(err_range) - 1):
            condition = (edp[:, 0] > err_range[i]) * (edp[:, 0] < err_range[i + 1])  # * (edp[:, 2] > 1)
            # condition = (edp[:, 0] > err_range[i]) * (edp[:, 0] < err_range[i + 1]) * (edp[:, 2] > 1)
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


def pos_count_fig(data, pos_count):
    fig, ax = plt.subplots(1, 1)
    for edp in data:
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


data = []
for file in files:
    edp = np.genfromtxt(file, delimiter=',')
    data.append(edp)
# all_pos_counts(files)
inp = []
for i in range(0, 20):
    for j in range(0, 20):
        inp.append((data, j, j + 1, i))

pool = Pool(20)
pool.map(pos_count_dist_fig, inp)
