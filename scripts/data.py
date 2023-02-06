from multiprocessing.pool import Pool

import numpy as np
import os

episode_duration = 1


class Config:
    def __init__(self):
        self.n_x = 100
        self.n_y = 100

        self.max_x = 52.5
        self.max_y = 34.

        self.n_dist = 100


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
        if ep[3] - ep[2] < episode_duration:
            continue
        for j in range(ep[0], ep[1] + 1 - episode_duration):
            ep_x = []
            ep_y = []
            for i in range(j, j + episode_duration):
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
    all_x = []
    all_y = []
    for ep in episodes:
        if ep[3] - ep[2] < episode_duration:
            continue
        for j in range(ep[0], ep[1] + 1 - episode_duration):
            ep_x = []
            ep_y = []
            for i in range(j, j + episode_duration):
                xy = data[i]
                xy = np.delete(xy, [0, 3, 4])
                x = np.array(xy)
                if np.random.uniform(0, 1) < 0.8 or True:
                    x[32] = -1
                    x[33] = -1
                y = xy[:][32:34]
                ep_x.append(x)
                ep_y.append(y)
            ep_x = np.array(ep_x)
            ep_y = np.array(ep_y[-1])
            all_x.append(ep_x)
            all_y.append(ep_y)
    return all_x, all_y


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
    for i in range(episode_duration):
        all_x[i] = []
    for ep in episodes:
        if ep[3] - ep[2] < episode_duration:
            continue
        for j in range(ep[0], ep[1] + 1 - episode_duration):
            ep_x = []
            ep_y = []
            for i in range(j, j + episode_duration):
                xy = data[i]
                xy = np.delete(xy, [0, 3, 4])
                x = np.array(xy)
                y = xy[:][32:34]
                ep_x.append(x)
                ep_y.append(y)
            ep_x = np.array(ep_x)
            ep_y = np.array(ep_y[-1])
            for i in range(episode_duration):
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
    for i in range(episode_duration):
        all_x[i] = []

    for ep in episodes:
        if ep[3] - ep[2] < episode_duration:
            continue
        for j in range(ep[0], ep[1] + 1 - episode_duration):
            ep_x = []
            ep_y = []
            for i in range(j, j + episode_duration):
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
            for i in range(episode_duration):
                p = ep_x[i][32], ep_x[i][33]
                new_ep = np.array(ep_x)
                new_ep[:, 32:34] = -1
                new_ep[i][32] = p[0]
                new_ep[i][33] = p[1]
                all_x[i].append(new_ep)
            all_y.append(ep_y)
    return all_x, all_y


def read_file(file_name):
    xy = np.genfromtxt(f'data/{file_name}', delimiter=',')[:, :-1]
    return xy


def read_file_test(file_name):
    xy = np.genfromtxt(f'data-test/{file_name}', delimiter=',')[1:, :]
    xy[:, 0] = np.round(xy[:, 0] * 6000)
    return create_episodes_dnn_test(xy)


def get_test_data():
    all_x: dict[int, list] = {}
    all_y = []
    for i in range(episode_duration):
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
    return all_x, all_y, episode_duration


def get_data(n=None):
    all_xy = []
    if n is not None:
        files = os.listdir('data/')[:n]
    else:
        files = os.listdir('data/')
    csv_files = []
    print('Reading-data...', end='')
    for file in files:
        if file.split('.')[-1] != 'csv':
            continue
        csv_files.append(file)
    pool = Pool(processes=20)
    res = pool.map(read_file, csv_files)
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
