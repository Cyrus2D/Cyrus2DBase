import multiprocessing
import os
import numpy as np

from models.config import config


def modify(file_name_index):
    print(f"#{file_name_index[1]}")
    file_name = file_name_index[0]
    xy = np.genfromtxt(f'{file_name}', delimiter=',')[:, :-1]
    xy = np.delete(xy, [185], axis=1)
    x = xy[:, 1:]
    print(x.shape)
    x1 = x.reshape((x.shape[0], 46, 8))
    for sample in x1:
        for obj in sample:
            if np.sum(np.isnan(obj)) == 8:
                obj[0] = -105
                obj[1] = -105
                obj[2] = 30
                obj[3] = -6
                obj[4] = -6
                obj[5] = 30
                obj[6] = -360
                obj[7] = 30
                continue
            if obj[5] == 1000 or obj[5] > 30:
                obj[3] = -6
                obj[4] = -6
                obj[5] = 30
            if obj[7] == 1000 or obj[7] > 30:
                obj[6] = -360
                obj[7] = 30
    xy[:, 1:] = x1.reshape((x.shape[0], 368))
    np.savetxt(f'{file_name}', xy, delimiter=',')


files = os.listdir('data/')
i = 0
all_files = []
for file in files:
    if file.split('.')[-1] != 'csv':
        continue
    i += 1
    all_files.append((f'data/{file}', i))
# config.n_process=20
# pool = multiprocessing.Pool(config.n_process)
# pool.map(modify, all_files)
