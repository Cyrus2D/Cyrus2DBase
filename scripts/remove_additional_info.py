import multiprocessing
import os
import numpy as np

from models.config import config


def modify(file_name_index):
    print(f"#{file_name_index[1]}")
    file_name = file_name_index[0]
    xy = np.genfromtxt(f'{file_name}', delimiter=',')[:, :-1]
    x = xy[:, 1:]
    x1 = x.reshape((x.shape[0], 45, 3))
    invalid = x1 == [-1, -1, -1]
    x2 = np.where(invalid, [-105, -105, 30], x1)
    x3 = x2.reshape((x.shape[0], 135))
    xy[:, 1:] = x3
    np.savetxt(f'{file_name_index[0]}', xy, delimiter=',')


files = os.listdir('data/')
i = 0
all_files = []
for file in files:
    if file.split('.')[-1] != 'csv':
        continue
    i += 1
    all_files.append((f'data/{file}', i))

pool = multiprocessing.Pool(config.n_process)
pool.map(modify, all_files)
