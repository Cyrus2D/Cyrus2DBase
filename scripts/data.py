from multiprocessing.pool import Pool

import numpy as np
import os
from models.config import config

def dist(x1, x2):
    return ((x1[:, 0] - x2[:, 0]) ** 2 + (x1[:, 1] - x2[:, 1]) ** 2) ** 0.5


def create_episodes_dnn(data):
    episodes = []
    last_cycle = None
    episode_start = None
    index_start = None
    for i in range(data.shape[0]):
        cycle = data[i][0]
        if last_cycle is None:
            last_cycle = cycle
            episode_start = cycle
            index_start = i
            continue

        if cycle - 1 != last_cycle:
            episodes.append((index_start, i - 1, episode_start, last_cycle))
            episode_start = cycle
            index_start = i
        last_cycle = cycle
    all_x = []
    all_y = []
    for ep in episodes:
        if ep[3] - ep[2] < config.episode_duration:
            continue
        for j in range(ep[0], ep[1] + 1 - config.episode_duration):
            ep_x = []
            ep_y = []
            for i in range(j, j + config.episode_duration):
                xy = data[i]
                xy = np.delete(xy, [0, 3, 4])
                x = np.array(xy)
                if np.random.uniform(0, 1) < 0.8:
                    x[32] = -1
                    x[33] = -1
                y = xy[:][32:34]
                ep_x.append(x)
                ep_y.append(y)
            ep_x = np.array(ep_x).flatten()
            ep_y = np.array(ep_y[-1])
            all_x.append(ep_x)
            all_y.append(ep_y)
    return all_x, all_y


def create_episodes_rnn(data):
    episodes = []
    last_cycle = None
    episode_start = None
    index_start = None
    for i in range(data.shape[0]):
        cycle = data[i][0]
        if last_cycle is None:
            last_cycle = cycle
            episode_start = cycle
            index_start = i
            continue

        if cycle - 1 != last_cycle:
            episodes.append((index_start, i - 1, episode_start, last_cycle))
            episode_start = cycle
            index_start = i
        last_cycle = cycle
    all_xy = []
    for ep in episodes:
        if ep[3] - ep[2] < config.episode_duration:
            continue
        for j in range(ep[0], ep[1] + 1 - config.episode_duration):
            ep_xy = []
            for i in range(j, j + config.episode_duration):
                ep_xy.append(data[i])
            all_xy.append(ep_xy)
    return all_xy


def create_episodes_dnn_test(data):
    episodes = []
    last_cycle = None
    episode_start = None
    index_start = None
    for i in range(data.shape[0]):
        cycle = data[i][0]
        if last_cycle is None:
            last_cycle = cycle
            episode_start = cycle
            index_start = i
            continue

        if cycle - 1 != last_cycle:
            episodes.append((index_start, i - 1, episode_start, last_cycle))
            episode_start = cycle
            index_start = i
        last_cycle = cycle
    all_x: dict[int, list] = {}
    all_y = []
    for i in range(config.episode_duration):
        all_x[i] = []
    for ep in episodes:
        if ep[3] - ep[2] < config.episode_duration:
            continue
        for j in range(ep[0], ep[1] + 1 - config.episode_duration):
            ep_x = []
            ep_y = []
            for i in range(j, j + config.episode_duration):
                xy = data[i]
                xy = np.delete(xy, [0, 3, 4])
                x = np.array(xy)
                y = xy[:][32:34]
                ep_x.append(x)
                ep_y.append(y)
            ep_x = np.array(ep_x)
            ep_y = np.array(ep_y[-1])
            for i in range(config.episode_duration):
                p = ep_x[i][32], ep_x[i][33]
                new_ep = np.array(ep_x)
                new_ep[:, 32:34] = -1
                # new_ep[i][32] = p[0]
                # new_ep[i][33] = p[1]
                all_x[i].append(new_ep.flatten())
            all_y.append(ep_y)
    return all_x, all_y


def create_episodes_rnn_test(data):
    episodes = []
    last_cycle = None
    episode_start = None
    index_start = None
    for i in range(data.shape[0]):
        cycle = data[i][0]
        if last_cycle is None:
            last_cycle = cycle
            episode_start = cycle
            index_start = i
            continue

        if cycle - 1 != last_cycle:
            episodes.append((index_start, i - 1, episode_start, last_cycle))
            episode_start = cycle
            index_start = i
        last_cycle = cycle

    all_x: dict[int, list] = {}
    all_y = []
    for i in range(config.episode_duration):
        all_x[i] = []

    for ep in episodes:
        if ep[3] - ep[2] < config.episode_duration:
            continue
        for j in range(ep[0], ep[1] + 1 - config.episode_duration):
            ep_x = []
            ep_y = []
            for i in range(j, j + config.episode_duration):
                xy = data[i]
                xy = np.delete(xy, [0, 3, 4])
                x = np.array(xy)
                # if np.random.uniform(0, 1) < 0.8:
                #     x[32] = -1
                #     x[33] = -1
                y = xy[:][32:34]
                ep_x.append(x)
                ep_y.append(y)

            ep_x = np.array(ep_x)
            ep_y = np.array(ep_y[-1])
            for i in range(config.episode_duration):
                p = ep_x[i][32], ep_x[i][33]
                new_ep = np.array(ep_x)
                new_ep[:, 32:34] = -1
                new_ep[i][32] = p[0]
                new_ep[i][33] = p[1]
                all_x[i].append(new_ep)
            all_y.append(ep_y)
    return all_x, all_y


def read_file(file_name_index):
    print(f'#{file_name_index[1]}')
    file_name = file_name_index[0]
    xy = np.genfromtxt(f'data/{file_name}', delimiter=',')[:, :]
    return xy


def read_file_rnn(file_name_index):
    print(f'#{file_name_index[1]}')
    file_name = file_name_index[0]
    xy = np.genfromtxt(f'data/{file_name}', delimiter=',')[:, :]
    return create_episodes_rnn(xy)


def read_file_test(file_name):
    xy = np.genfromtxt(f'data-test/{file_name}', delimiter=',')[1:, :]
    xy[:, 0] = np.round(xy[:, 0] * 6000)
    return create_episodes_dnn_test(xy)


def get_test_data():
    all_x: dict[int, list] = {}
    all_y = []
    for i in range(config.episode_duration):
        all_x[i] = []
    files = os.listdir('data-test/')[:100]
    csv_files = []
    print('Reading-data...', end='')
    for file in files:
        if file.split('.')[-1] != 'csv':
            continue
        csv_files.append(file)
    pool = Pool(processes=20)
    res = pool.map(read_file_test, csv_files)
    for r in res:
        for k, v in r[0].items():
            all_x[k] += v
        all_y += r[1]
    print('Done!')
    return all_x, all_y, config.episode_duration


def get_data(n=None, m=None):
    all_xy = []
    if n is not None:
        files = os.listdir('data/')[:n]
    elif m is not None:
        files = os.listdir('data/')[-m:]
    else:
        files = os.listdir('data/')
    csv_files = []
    print('Reading-data...', end='')
    i = 0
    for file in files:
        if file.split('.')[-1] != 'csv':
            continue
        i += 1
        csv_files.append((file, i))
    pool = Pool(processes=20)
    res = pool.map(read_file, csv_files)
    for r in res:
        all_xy += list(r)
    print('Done!')
    return all_xy


def get_data_rnn(n=None, m=None):
    all_xy = []
    if n is not None:
        files = os.listdir('data/')[:n]
    elif m is not None:
        files = os.listdir('data/')[-m:]
    else:
        files = os.listdir('data/')
    csv_files = []
    print('Reading-data...', end='')
    i = 0
    for file in files:
        if file.split('.')[-1] != 'csv':
            continue
        i += 1
        csv_files.append((file, i))
    pool = Pool(processes=20)
    res = pool.map(read_file_rnn, csv_files)
    for r in res:
        all_xy += list(r)
    print('Done!')
    return all_xy


def create_headers():
    headers = {}
    headers['cycle'] = [0]
    headers['ball'] = [1, 2, 3]

    # 4 5 6, 7 8 9
    for i in range(1, 12):
        headers[f'tm-{i}-noise'] = list(range(4 + (i - 1) * 3, 4 + i * 3))  # max=4+11*3 = 37
        headers[f'opp-{i}-noise'] = list(range(37 + (i - 1) * 3, 37 + i * 3))  # max = 37+11*3 = 33+37 = 70
        headers[f'tm-{i}-full'] = list(range(70 + (i - 1) * 3, 70 + i * 3))  # max=70 + 33 = 103
        headers[f'opp-{i}-full'] = list(range(103 + (i - 1) * 3, 103 + i * 3))

    return headers


def create_x_y_indexes(headers: dict[str, list[int]]):
    x_indexes = []
    y_indexes = []
    for key, value in headers.items():
        if key in ['cycle']:
            continue
        if key.find('full') != -1:
            continue
        x_indexes += value
        print(f"x {key}")
    print(x_indexes)

    for key, value in headers.items():
        if key in ['cycle']:
            continue
        if key.find('noise') != -1:
            continue
        if key.find('ball') != -1:
            continue
        if key.find('tm') != -1:
            continue
        y_indexes += value[:-1]
        print(f"y {key}")
    print(y_indexes)

    return x_indexes, y_indexes


def create_labeled_y(xy, n_label, r):
    headers = create_headers()

    opp_pos_noise = np.array(xy[:, headers['opp-5-noise']][:, :-1])
    opp_pos_full = xy[:, headers['opp-5-full']][:, :-1]

    opp_pos_noise = np.where(opp_pos_noise == [-105, -105], [0, 0], opp_pos_noise)

    opp_err = opp_pos_full - opp_pos_noise
    opp_err = np.clip(opp_err, -r / 2, r / 2)
    index = np.floor((opp_err / r) * (n_label - 1)) + n_label / 2
    y_index = np.array(index[:, 0] * n_label + index[:, 1], dtype=np.uint32)

    y = np.zeros((opp_pos_noise.shape[0], n_label ** 2))
    y[np.arange(y_index.size), y_index] = 1

    r_indexes = np.arange(y.shape[0])
    np.random.shuffle(r_indexes)
    for i in r_indexes[:5]:
        print(opp_pos_noise[i])
        print(opp_pos_full[i])
        print(opp_err[i])
        print(index[i])
        print(y_index[i])
        print(np.argmax(y[i]))
    return y


def normalize_data(x, y=None):
    pos_x_i = [i for i in range(0, 69, 3)]
    pos_y_i = [i for i in range(1, 69, 3)]
    pos_count_i = [i for i in range(2, 69, 3)]

    x[:, pos_x_i] /= config.max_x
    x[:, pos_y_i] /= config.max_y
    x[:, pos_count_i] /= 30
    if y is not None:
        y[:, 0] /= config.max_x
        y[:, 1] /= config.max_y


def normalize_data_all(x, y=None):
    pos_x_i = [i for i in range(0, 69, 3)]
    pos_y_i = [i for i in range(1, 69, 3)]
    pos_count_i = [i for i in range(2, 69, 3)]

    x[:, pos_x_i] /= config.max_x
    x[:, pos_y_i] /= config.max_y
    x[:, pos_count_i] /= 30.

    pos_x_i = [i for i in range(0, 22, 2)]
    pos_y_i = [i for i in range(1, 22, 2)]

    if y is not None:
        y[:, pos_x_i] /= config.max_x
        y[:, pos_y_i] /= config.max_y


def normalize_data_rnn_all(x, y=None):
    pos_x_i = [i for i in range(0, 69, 3)]
    pos_y_i = [i for i in range(1, 69, 3)]
    pos_count_i = [i for i in range(2, 69, 3)]

    x[:, :, pos_x_i] /= config.max_x
    x[:, :, pos_y_i] /= config.max_y
    x[:, :, pos_count_i] /= 30.

    pos_x_i = [i for i in range(0, 22, 2)]
    pos_y_i = [i for i in range(1, 22, 2)]

    if y is not None:
        y[:, pos_x_i] /= config.max_x
        y[:, pos_y_i] /= config.max_y


def normalize_data_rnn(x, y=None):
    pos_x_i = [i for i in range(0, 69, 3)]
    pos_y_i = [i for i in range(1, 69, 3)]
    pos_count_i = [i for i in range(2, 69, 3)]

    x[:, :, pos_x_i] /= config.max_x
    x[:, :, pos_y_i] /= config.max_y
    x[:, :, pos_count_i] /= 30
    if y is not None:
        y[:, 0] /= config.max_x
        y[:, 1] /= config.max_y
