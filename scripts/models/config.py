class Config:
    def __init__(self):
        self.n_x = 100
        self.n_y = 100

        self.max_x = 52.5
        self.max_y = 34.

        self.n_dist = 20

        self.n_inputs = 8 + 11 * 2 * 8
        self.n_outputs = 11 * 2

        self.optimizer = 'adam'
        self.loss = 'mse'

        self.batch_size = 64
        self.n_epochs = 7

        self.n_train_file = 300
        self.n_test_file = 100

        self.res_dir = "res"

        self.episode_duration = 10
        self.n_process = 100


config = Config()
