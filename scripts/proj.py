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
    headers, _ = create_headers() # get dictionary of headers and their indexs

    print('DNN')
    xy_train = np.array(get_data(config.n_train_file)) # creating DNN data for train
    xy_test = np.array(get_data(m=config.n_test_file)) # creating DNN data for test

    x_indexes, y_indexes = create_x_y_indexes(headers) # find input headers and output headers

    # seperate the date to X and Y
    x = np.array(xy_train[:, x_indexes]) 
    y = np.array(xy_train[:, y_indexes])

    print('normalizing')
    normalize_data_all(x, y) # normalizing data

    # shuffling data
    r_indexes = np.arange(x.shape[0])
    np.random.shuffle(r_indexes)

    print('shuffling')
    x = x[r_indexes]
    y = y[r_indexes]

    # seperating test data for etst
    xt = np.array(xy_test[:, x_indexes])
    # normalizing test data
    normalize_data_all(xt)

    # Initiate different DNN Models
    model = [
        DNN_Model([128, 64], ['relu', 'relu']),
        DNN_Model([256, 128], ['relu', 'relu']),
        DNN_Model([512, 256], ['relu', 'relu']),
        DNN_Model([512, 256, 128, 64, 32], ['relu', 'relu', 'relu', 'relu', 'relu']),
        DNN_Model([256, 128, 64, 32], ['relu', 'relu', 'relu', 'relu']),
    ]

    for m in model: # loop through the models
        print(m.get_name()) 
        m.fit(x, y, headers) # fit the train data
        m.test(xy_test, xt, headers) # test and save the error of test data

    # loop through different episode_durations (lookback period)
    for episode_duration in [5, 10, 15]:
        print(f'LSTM-{episode_duration}')

        xy_train = np.array(get_data_rnn(config.n_train_file, episode_duration=episode_duration)) # Creating train data based on episode duration
        xy_test = np.array(get_data_rnn(m=config.n_test_file, episode_duration=episode_duration)) # Creating test data based on episode duration

        x_indexes, y_indexes = create_x_y_indexes(headers)

        # seperating data
        x = np.array(xy_train[:, :, x_indexes])
        y = np.array(xy_train[:, -1, y_indexes])

        print('normalizing')
        normalize_data_rnn_all(x, y) # normaling data

        # Shuffling data
        r_indexes = np.arange(x.shape[0])
        np.random.shuffle(r_indexes)

        print('shuffling')
        x = x[r_indexes]
        y = y[r_indexes]

        xt = np.array(xy_test[:, :, x_indexes])
        normalize_data_rnn_all(xt)

        # Initializing LSTM models with specific episode duration
        model = [
            LSTM_Model([256, 128], ['relu', 'relu'], episode_duration),
            LSTM_Model([512, 256], ['relu', 'relu'], episode_duration),
            LSTM_Model([128, 64, 32], ['relu', 'relu', 'relu'], episode_duration),
            LSTM_Model([512, 256, 128, 32], ['relu', 'relu', 'relu', 'relu'], episode_duration),
        ]

        # fit the models, test them and save the results.
        for m in model:
            print(m.get_name())
            m.fit(x, y, headers)
            m.test(xy_test, xt, headers)


def dist(x1, x2):
    return ((x1[:, 0] - x2[:, 0]) ** 2 + (x1[:, 1] - x2[:, 1]) ** 2) ** 0.5


def make_last_seen_error():
    # config.n_process = 20
    headers, _ = create_headers()
    # read test files
    xy = np.array(get_data(m=config.n_test_file))

    my_pos = (xy[:, headers["tm-9-full"]])[:, :2] # find observer pos
    opp_pos_noise = (xy[:, headers["opp-5-noise"]])[:, :2] # find noisy (last-seen) opponent pos (player 5 is test subject) 
    opp_pos_full = (xy[:, headers["opp-5-full"]])[:, :2] #  find accurate opponent pos

    error = dist(opp_pos_noise, opp_pos_full) # calculate the distance error
    my_dist = dist(my_pos, opp_pos_full) # calculate the distance between observer and test subject
    pos_count = (xy[:, headers["opp-5-noise"]])[:, 2] # check the pos count of the player

    all = np.zeros((error.shape[0], 3)) # tmp array to concatinate the data
    all[:, 0] = error
    all[:, 1] = my_dist
    all[:, 2] = pos_count

    np.savetxt(f"res/edp-data", all, delimiter=',') # save the data


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
    print("Comapring Heatmap Figs: ", args[2], args[3])
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
    pos_count_dist_2 = np.where((counter_1 < 3) * (counter_2 < 3), np.nan, pos_count_dist_2)
    pos_count_dist_1 = np.where((counter_1 < 3) * (counter_2 < 3), np.nan, pos_count_dist_1)

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
    # im = ax.imshow(Z, cmap='bwr'), vmin=-5, vmax=+5)
    im = ax.imshow(Z, cmap=cmap, vmin=-v, vmax=+v)
    ax.figure.colorbar(im, ax=ax, shrink=0.5)
    fig.tight_layout()
    ax.set_xlabel("pos-count")
    ax.set_ylabel("dist")

    # plt.show()
    plt.savefig(f"res/compare/E-{f1}-vs-E-{f2}") # saving heatmap figur
    plt.close()
    # surf = ax.plot_surface(X, Y, Z, facecolors=c, antialiased=False)

    # pickle.dump(fig, open('figs/accuracy.pickle', 'wb'))
    # plt.show()


def make_heat_maps():
    file_list = os.listdir('res/')
    files = [ # make a list of test errors files
        f'res/{file}' for file in file_list if
        (file.startswith('edp') )
    ]

    # read the test errors files
    data = []
    for file in files:
        print(file)
        edp = np.genfromtxt(file, delimiter=',')
        data.append(edp)
    
    # comaper one by one and create heat map
    for i in range(len(data)):
        for j in range(len(data)):
            compare_heat_map((data[i], data[j], files[i], files[j]))


def pos_count_fig(data, files, pos_count):
    fig, ax = plt.subplots(1, 1)
    
    # looping through different test errors of each model
    for edp, file in zip(data, files):
        
        counter = [] # a list to save the number of test-errors in different distance ranges
        
        # loop through differrent distance changes
        for i in range(len(err_range) - 1):
            # create the condition to count the number  of test error cases
            
            condition = ((edp[:, 0] > err_range[i])         # the dist error be more than begining of the dist-range    and
                              * (edp[:, 0] < err_range[i + 1])   # the dist error be less than end of the dist-range         and
                              * (edp[:, 2] == pos_count))        # the test case poscount be equal to the specified test case
            
            counter.append(np.sum(np.where(condition, 1, 0)))    # sum the number of test cases with the condition
            # counter.append(np.sum(np.where(condition, 1, 0))
            #                + (np.sum(counter[-1] if i != 0 else 0)))
        # add a dummy 0
        counter.append(0)
        counter = np.array(counter) # arraying
        # print(counter / counter[-1])
        
        # coloring
        if file.find('lstm') != -1:
            color = 'red'
        elif file.find('rnn') != -1:
            color = 'blue'
        elif file.find('dnn') != -1:
            color = 'green'
        else:
            color = 'blue'
        # ploting
        ax.plot(err_range[:100], counter[:100] / np.sum(counter[:100]), c=color, label=file)
        # ax.plot(err_range, counter / np.sum(counter), color=color, label=file)

    ax.legend()
    plt.title(f"pc={pos_count}")
    plt.savefig(f'res/pc/{pos_count}.png') # saving
    # plt.show()
    plt.close()


def draw_2d_based_on_poscount():
    file_list = os.listdir('res/')
    # files = [
    #     f'res/{file}' for file in file_list if
    #     (file.startswith('edp') and file != 'edp-data')
    # ]

    files = [ # making file list of best models and data(last-seen) 
        'res/edp-data', 
        'res/edp-dnn-512-256-128-64-32-relu-relu-adam-mse-64',
        'res/edp-lstm-512-256-relu-relu-adam-mse-64-5'
    ]
    data = []
    # Reading data of files
    for file in files:
        print(file)
        edp = np.genfromtxt(file, delimiter=',')
        data.append(edp)

    # Creating 2D comparison(error, dist) with different poscounts
    for i in range(20):
        pos_count_fig(data, files, i)


def compare3d(args):
    print("3D Compare Fig: ", args[2], args[3])
    edp1 = args[0] # get test error of model 1
    edp2 = args[1] # get test error of model 2
    f1 = args[2] # model 1 name
    f2 = args[3] # model 2 name
    f1 = f1.split('/')[-1]
    f2 = f2.split('/')[-1]
    
    # creating zero matrix for summing up the error based on distance and poscount
    pos_count_dist_1 = [[0 for _ in range(int(30 + 1))] for _ in range(int(config.n_dist + 1))]
    # creating zero matrix for counting the number of each cell for averaging
    counter_1 = [[0 for _ in range(int(30 + 1))] for _ in range(int(config.n_dist + 1))]

    max_dist = max(np.max(edp1[:, 1]), np.max(edp2[:, 1]))# finding max dist of errors

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
    pos_count_dist_2 = np.where((counter_1 < 3) * (counter_2 < 3), np.nan, pos_count_dist_2)
    pos_count_dist_1 = np.where((counter_1 < 3) * (counter_2 < 3), np.nan, pos_count_dist_1)

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
    ax.set_xlabel("pos-count")
    ax.set_ylabel("dist")
    ax.set_zlabel("error")

    # plt.show()
    # plt.savefig(f"res/compare/E-{f1}-vs-E-{f2}")
    # plt.close()
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, antialiased=False) # draw 3d surface
    plt.savefig(f"res/compare3d/E-{f1}-vs-E-{f2}") # saving fig
    plt.close()

    # pickle.dump(fig, open('figs/accuracy.pickle', 'wb'))
    # plt.show()


def make_3d():
    file_list = os.listdir('res/')
    files = [ # make a list of test errors
        f'res/{file}' for file in file_list if
        (file.startswith('edp'))
    ]

    data = []
    # read the data of test erroes
    for file in files:
        print(file)
        edp = np.genfromtxt(file, delimiter=',')
        data.append(edp)
    
    # make 3d figures for comparison
    for i in range(len(data)):
        for j in range(len(data)):
            compare3d((data[i], data[j], files[i], files[j]))


if __name__ == "__main__":
    make_last_seen_error() # Making last-seen errors
    train_and_test_models() # Train the models and test them and save the test errors
    make_heat_maps() # create comparing heat map
    make_3d() # create 3d comaprison figures
    draw_2d_based_on_poscount() # darw 2d error-distance with different poscounts
