class Config:
    def __init__(self):
        self.n_x = 100
        self.n_y = 100

        self.max_x = 52.5 # field half length
        self.max_y = 34. # field half width

        self.n_dist = 20 # quantize the distance to number of n_dist

        self.n_inputs = 8 + 11 * 2 * 8 # input feature number
        self.n_outputs = 11 * 2 # output feature number

        self.optimizer = 'adam' # optimizer
        self.loss = 'mse' # loss function

        self.batch_size = 64 # batch size
        self.n_epochs = 1 # number of epochs for each model

        self.n_train_file = 1 # number of train files to read and train
        self.n_test_file = 1 # number of test fiels to read and test

        self.res_dir = "res" # result directory

        self.episode_duration = 10 # episode duration for lstm
        self.n_process = 5 # number of process to create for reading files of data


config = Config()
