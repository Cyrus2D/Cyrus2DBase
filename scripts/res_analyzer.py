import os
from multiprocessing.pool import Pool

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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
    return error


# data_error()
# exit()

file_list = os.listdir('res/')

files = [
    f'res/{file}' for file in file_list if
    (file.startswith('edp'))
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
# files.append('res/edp-data')


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

def compare_3d(args):
    edp1 = args[0] # errors of model 1
    edp2 = args[1] # errors of model 2
    f1 = args[2]  # name of model 1
    f2 = args[3]  # name of model 2
    f1 = f1.split('/')[-1]
    f2 = f2.split('/')[-1]
    
    # creating zero matrix for summing up the error based on distance and poscount
    pos_count_dist_1 = [[0 for _ in range(int(30 + 1))] for _ in range(int(config.n_dist + 1))]
    # creating zero matrix for counting the number of each cell for averaging
    counter_1 = [[0 for _ in range(int(30 + 1))] for _ in range(int(config.n_dist + 1))]

    max_dist = max(np.max(edp1[:, 1]), np.max(edp2[:, 1])) # finding max dist of errors
    # filling the pos_count_dist_1 for model 1
    for i in range(edp1.shape[0]): 
        pc = int(edp1[i][2]) # find poscount index in the matrix
        d = int((edp1[i][1] / max_dist) * config.n_dist) # finding distance index in the matrix
        e = edp1[i][0] # get the error of the model in the test case
        # e = edp1[i][0] / edp1[i][1]

        pos_count_dist_1[d][pc] += e # adding the error to matrix
        counter_1[d][pc] += 1 # increasing the counter
    pos_count_dist_1 = np.array(pos_count_dist_1) # arraying
    counter_1 = np.array(counter_1) # arraying

    # replacing zero with one in counter matrix to prohibit the dividing by 0
    for i in range(counter_1.shape[0]):
        for j in range(counter_1.shape[1]):
            counter_1[i][j] = 1 if counter_1[i][j] == 0 else counter_1[i][j]

    # doing the same procedure for model 2
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

    # removing data where there is not enough data to comaper (3 is threshhold)
    pos_count_dist_2 = np.where((counter_1 < 100) * (counter_2 < 100), np.nan, pos_count_dist_2)
    pos_count_dist_1 = np.where((counter_1 < 100) * (counter_2 < 100), np.nan, pos_count_dist_1)

    # averaging
    pos_count_dist_1 /= counter_1
    pos_count_dist_2 /= counter_2

    # making X and Y and Z values to create the heat map
    Y = np.arange(0, max_dist, max_dist / (config.n_dist + 1))
    X = np.arange(0, 30 + 1, 1.)[:-5]
    X, Y = np.meshgrid(X, Y)
    Z = (pos_count_dist_2 - pos_count_dist_1)[:, :-5]

    min_z = np.nanmin(Z)
    max_z = np.nanmax(Z)
    v = max(max_z, abs(min_z))

    fig, ax = plt.subplots()
  
    # add color to the fig, 
    # LSTM is Red
    # DNN is Green
    # blue for data (last-seen)
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

    # create color-map
    cmap = get_cmp(first_color, second_color)

    # create black background to cover cells that has insufficeint data
    ZB = np.where((counter_1 < 100) * (counter_2 < 100), 1, 0)
    im = ax.imshow(ZB, cmap='Greys', vmin=0, vmax=+1)
    im = ax.imshow(Z, cmap=cmap, vmin=-v, vmax=+v)
    ax.figure.colorbar(im, ax=ax, shrink=0.5)
    fig.tight_layout()
    ax.set_xlabel("pos-count")
    ax.set_ylabel("dist")

    plt.savefig(f"res/compare/E-{f1}-vs-E-{f2}") # saving heatmap figur
    plt.close()

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
    for j in range(len(data)):
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
