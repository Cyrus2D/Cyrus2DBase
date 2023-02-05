from multiprocessing.pool import Pool

import numpy as np
import os

episode_duration = 10


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
                x = np.delete(xy, [32, 33])
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
                if np.random.uniform(0, 1) < 0.8:
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
                ep_x[:, 32:34] = -1
                ep_x[i][32] = p[0]
                ep_x[i][33] = p[1]
                all_x[i].append(ep_x)
            all_y.append(ep_y)
    return all_x, all_y


def read_file(file_name):
    xy = np.genfromtxt(f'data/{file_name}', delimiter=',')[1:, :]
    xy[:, 0] = np.round(xy[:, 0] * 6000)
    return create_episodes_rnn(xy)


def read_file_test(file_name):
    xy = np.genfromtxt(f'data-test/{file_name}', delimiter=',')[1:, :]
    xy[:, 0] = np.round(xy[:, 0] * 6000)
    return create_episodes_rnn_test(xy)


def get_test_data():
    all_x = []
    all_y = []
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
        all_x += r[0]
        all_y += r[1]
    print('Done!')
    return all_x, all_y, episode_duration


def get_data():
    all_x = []
    all_y = []
    files = os.listdir('data/')[:100]
    csv_files = []
    print('Reading-data...', end='')
    for file in files:
        if file.split('.')[-1] != 'csv':
            continue
        csv_files.append(file)
    pool = Pool(processes=20)
    res = pool.map(read_file, csv_files)
    for r in res:
        all_x += r[0]
        all_y += r[1]
    print('Done!')
    return all_x, all_y, episode_duration
