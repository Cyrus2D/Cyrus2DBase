from multiprocessing.pool import Pool

import numpy as np
import os
from models.config import config


# Create Data Series for LSTM/RNN
# data is the numpy array that is created from the csv file
# episode duration is the look back period
def create_episodes_rnn(data, episode_duration):
    episodes = []
    last_cycle = None
    episode_start = None
    index_start = None
    
    # loop through the whole game, find the State Sequences in the game
    for i in range(data.shape[0]):
        cycle = data[i][0]
        if last_cycle is None:
            last_cycle = cycle
            episode_start = cycle
            index_start = i
            continue

        if cycle - 1 != last_cycle: # finding the end of a state sequence / End of PLAY ON game mode
            episodes.append((index_start, i - 1, episode_start, last_cycle)) # saving the indexes of  state sequence in the list
            episode_start = cycle
            index_start = i
        last_cycle = cycle

    all_xy = []
    # loop through state sequences to create data for lstm/rnn  
    for ep in episodes:
        if ep[3] - ep[2] < episode_duration: # if the state sequen is less than epsiode duration(look back period), do not consider it
            continue
        # otherwise, move a window frame step by step through the state sequence and create new series of data for lstm/rnn
        for j in range(ep[0], ep[1] + 1 - episode_duration):
            ep_xy = [] # create a list to save a sample data
            for i in range(j, j + episode_duration):
                ep_xy.append(data[i])  # add to sample data
            all_xy.append(ep_xy) # save the sample data in the data set
    return all_xy



# read csv file to create data for dnn (no series of data/ only a single state)
def read_file(file_name_index):
    file_name = file_name_index[0]
    xy = np.genfromtxt(f'data/{file_name}', delimiter=',')[:, :] # get array data from csv
    return xy


# read csv file to ceate the data series for lstm/rnn.
def read_file_rnn(file_name_index):
    file_name = file_name_index[0]
    xy = np.genfromtxt(f'data/{file_name}', delimiter=',')[:, :]
    return create_episodes_rnn(xy, file_name_index[2])


# get the whole data set folder, loop through all of them and read them in parallel mode to make data for DNN
# if n is a number and m is none -> this function starts reading n nubmer of file from the begining of the list
# if m is a number and n is none -> this function starts reading m nubmer of file from the end of the list
def get_data(n=None, m=None):
    all_xy = []
    
    # making a list of csv files to read
    if n is not None:
        files = os.listdir('data/')[:n]
    elif m is not None:
        files = os.listdir('data/')[-m:]
    else:
        files = os.listdir('data/')
    csv_files = []
    print('Reading-data...', end='')
    i = 0
    
    # loop through fiels and create input data for functions (to use in parallel mode)
    for file in files:
        if file.split('.')[-1] != 'csv':
            continue
        i += 1
        csv_files.append((file, i)) # save inputs in a list
    pool = Pool(processes=config.n_process) # create a Pool for multi processing purpose
    res = pool.map(read_file, csv_files) # start the pool and save the out put in a list
    for r in res:
        all_xy += list(r) # concate the whole data in a list
    print('Done!')
    return all_xy


# get the whole data set folder, loop through all of them and read them in parallel mode to make data-series for LSTM/RNN
# if n is a number and m is none -> this function starts reading n nubmer of file from the begining of the list
# if m is a number and n is none -> this function starts reading m nubmer of file from the end of the list
def get_data_rnn(n=None, m=None, episode_duration=0):
    all_xy = []
    
    # create a list of csv files
    if n is not None:
        files = os.listdir('data/')[:n]
    elif m is not None:
        files = os.listdir('data/')[-m:]
    else:
        files = os.listdir('data/')
    csv_files = []
    print('Reading-data...', end='')
    i = 0
    
    # preparing input arguments for the function to pass them in parallel mode
    for file in files:
        if file.split('.')[-1] != 'csv':
            continue
        i += 1
        csv_files.append((file, i, episode_duration))
    pool = Pool(processes=config.n_process) # make the pool for multiprocessing procedure
    res = pool.map(read_file_rnn, csv_files) # start the pool and get the return values as a list
    for r in res:
        all_xy += list(r) # concat the whole readed data
    print('Done!')
    return all_xy


# this function creates HEADERS for the data to access specific data when it is required.
# The header is a dectionary
# key is the header name
# value is a list of indexs that the header include them.
def create_headers():
    headers = {}
    headers['cycle'] = [0] # cycle header and its index
    headers['ball'] = [1, 2, 3, 4, 5, 6, 7, 8] # ball header and its features indexes

    # 4 5 6, 7 8 9
    for i in range(1, 12):
        # each player has 8 different features
        # using the order of the data
        # we loop through the player numbers and save a list of their indexes
        
        # noise header is the noisy-data
        # full header is the accurate data of each player in the field
        # tm means teammates
        # opp means opponents
        headers[f'tm-{i}-noise'] = list(range(9 + (i - 1) * 8, 9 + i * 8))  # max=4+11*3 = 37
        headers[f'opp-{i}-noise'] = list(range(97 + (i - 1) * 8, 97 + i * 8))  # max = 37+11*3 = 33+37 = 70
        headers[f'tm-{i}-full'] = list(range(8 + 185 + (i - 1) * 8, 8 + 185 + i * 8))  # max=70 + 33 = 103
        headers[f'opp-{i}-full'] = list(range(8 + 273 + (i - 1) * 8, 8 + 273 + i * 8))

    sub_headers = {
        'pos': [0, 1, 2],
        'vel': [3, 4, 5],
        'body': [6, 7]
    }
    return headers, sub_headers

# create x y indexes means the creating the index of the input and output of the model
# it returns a tuple containing two list
# first list contains the input indexes of the whole data for the models
# second list is the output indexes of the whold data for the models
def create_x_y_indexes(headers: dict[str, list[int]]):
    x_indexes = []
    y_indexes = []
    
    # loop through headers for x indexes
    for key, value in headers.items():
        if key in ['cycle']: # remove cycle info
            continue
        if key.find('full') != -1: # remove full/accurate info of objects
            continue
        x_indexes += value # add others (noisy information)

    # loop through headers for y indexes
    for key, value in headers.items():
        if key in ['cycle']: # remove cycle
            continue
        if key.find('noise') != -1: # remove noisy info
            continue
        if key.find('ball') != -1: # remove ball info
            continue
        if key.find('tm') != -1: # remove team-mates info
            continue
        y_indexes += value[:2] # add other objects' accurate position(only the accurate position of the opponents)

    return x_indexes, y_indexes


# normalize data with type of DNN input (no data series/ only samples)
# inputs are x and y (inpuys and outputs of the model)
# y can be None because for testing models we dont need to normalize y
def normalize_data_all(x, y=None):
    # Get the indexs of different features with same domain range.
    pos_x_i = [i for i in range(0, config.n_inputs, 8)] # x of positoins
    pos_y_i = [i for i in range(1, config.n_inputs, 8)] # y of positions
    pos_count_i = [i for i in range(2, config.n_inputs, 8)] # poscounts
    vel_x_i = [i for i in range(3, config.n_inputs, 8)] # x of velocity
    vel_y_i = [i for i in range(4, config.n_inputs, 8)] # y of velocity
    vel_count_i = [i for i in range(5, config.n_inputs, 8)] # vel count
    body_i = [i for i in range(6, config.n_inputs, 8)]      # body angle
    body_count_i = [i for i in range(7, config.n_inputs, 8)]#  body count

    # normalize the data based on their MAX and Min.
    x[:, pos_x_i] /= config.max_x
    x[:, pos_y_i] /= config.max_y
    x[:, pos_count_i] /= 30.
    x[:, vel_x_i] /= 3
    x[:, vel_y_i] /= 3
    x[:, vel_count_i] /= 30
    x[:, body_i] /= 180
    x[:, body_count_i] /= 30

    # make indexes for y 
    pos_x_i = [i for i in range(0, 22, 2)]
    pos_y_i = [i for i in range(1, 22, 2)]

    # normalize the output data if y is not none 
    if y is not None:
        y[:, pos_x_i] /= config.max_x
        y[:, pos_y_i] /= config.max_y


# normalize data with type of LSTM/RNN input (data series)
# inputs are x and y (inpuys and outputs of the model)
# y can be None because for testing models we dont need to normalize y
def normalize_data_rnn_all(x, y=None):
    # Get the indexs of different features with same domain range.
    pos_x_i = [i for i in range(0, config.n_inputs, 8)] # x of positoins
    pos_y_i = [i for i in range(1, config.n_inputs, 8)] # y of positions
    pos_count_i = [i for i in range(2, config.n_inputs, 8)] # poscounts
    vel_x_i = [i for i in range(3, config.n_inputs, 8)] # x of velocity
    vel_y_i = [i for i in range(4, config.n_inputs, 8)] # y of velocity
    vel_count_i = [i for i in range(5, config.n_inputs, 8)] # vel count
    body_i = [i for i in range(6, config.n_inputs, 8)]      # body angle
    body_count_i = [i for i in range(7, config.n_inputs, 8)]  # body count

    # normalize the data based on their MAX and Min.
    x[:, :, pos_x_i] /= config.max_x
    x[:, :, pos_y_i] /= config.max_y
    x[:, :, pos_count_i] /= 30.
    x[:, :, vel_x_i] /= 3
    x[:, :, vel_y_i] /= 3
    x[:, :, vel_count_i] /= 30
    x[:, :, body_i] /= 180
    x[:, :, body_count_i] /= 30

    # make indexes for y 
    pos_x_i = [i for i in range(0, 22, 2)]
    pos_y_i = [i for i in range(1, 22, 2)]

    # normalize the output data if y is not none 
    if y is not None:
        y[:, pos_x_i] /= config.max_x
        y[:, pos_y_i] /= config.max_y

def dist(x1, x2):
    return ((x1[:, 0] - x2[:, 0]) ** 2 + (x1[:, 1] - x2[:, 1]) ** 2) ** 0.5
