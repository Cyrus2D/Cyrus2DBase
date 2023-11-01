from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module
from data import create_x_y_indexes, normalize_data_rnn_all, dist
from models.config import config
import torch.optim as optim

activation_str_dict = {
    'relu': nn.ReLU,
    'softmax': nn.Softmax,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid
}
optimizer_str_dict = {

    'adam': optim.Adam,
    'sgd': optim.SGD,
    'rmsprop': optim.RMSprop
}
loss_str_dict = {
    'mse': nn.MSELoss,
    'crossentropy': nn.CrossEntropyLoss
}


class TorchLSTM(Module):

    def __init__(self, n_layers=[128, 64], activation=['relu', 'relu']):
        super(TorchLSTM, self).__init__()
        if len(n_layers) != len(activation):
            raise ValueError("You must give one and only one activation functuon for each layer!")
        self.n_layers = n_layers
        self.activations = activation
        layers_dict = OrderedDict()
        layers_dict['lstm'] = nn.LSTM(config.n_inputs, n_layers[0], dtype=torch.float64, batch_first=False)
        for i in range(len(n_layers) - 1):
            layers_dict[f'linear{i}'] = nn.Linear(n_layers[i], n_layers[i + 1], dtype=torch.float64)
            layers_dict[f'activation{i}'] = activation_str_dict[activation[i]]()
        layers_dict['linear_final'] = nn.Linear(n_layers[-1], 400, dtype=torch.float64)
        for k, v in layers_dict.items():
            self.add_module(k, v)

        self.layers_dict = layers_dict

        self.optimizer = optimizer_str_dict[config.optimizer](self.parameters(), lr=0.001)
        self.loss_fn = nn.CrossEntropyLoss()
        # self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')

    def get_name(self, prefix=''):
        return f"{config.res_dir}/torch/{prefix}lstm-{'-'.join(map(str, self.n_layers))}-{'-'.join(self.activations)}-{config.optimizer}-{config.loss}-{config.batch_size}-{config.episode_duration}"

    def forward(self, x):
        # do first LSTM layer
        # h_0 = torch.zeros( x.shape[0], self.n_layers[0], dtype=torch.float64).to(self.device)
        # c_0 = torch.zeros( x.shape[0], self.n_layers[0], dtype=torch.float64).to(self.device)

        lstm = self.get_submodule('lstm').to(self.device)
        output, (h_n, c_n) = lstm(x,)
        # print("output shape", output.shape)
        # print("h_n shape", h_n.shape)
        # print("c_n shape", c_n.shape)
        # output = h_n
        # output = output.view(output.shape[0], 1, output.shape[1])
        # print("output shape", output.shape)
        # output = output.view(-1, self.n_layers[0])
        output = output[:, -1, :]
        # do the rest
        for k, layer in self.layers_dict.items():
            if k == 'lstm':
                continue
            output = layer(output)
            # print(f'{k} output shape', output.shape)
        output = nn.functional.softmax(output, dim=1)
        return output

    def train(self, x_in, y_in, headers):
        print(self)
        # print("x shape", x_in.shape)
        x: torch.Tensor = torch.from_numpy(x_in).to(self.device)
        y = torch.from_numpy(y_in).to(self.device)
        # cast x,y to float64
        x = x.type(torch.float64)
        y = y.type(torch.float64)
        # print(x.dtype)
        epochs = config.n_epochs
        batch_size = config.batch_size
        # train validation split
        # print("x shape", x.shape)
        # print("y shape", y.shape)
        validation_split = 0.1
        train_size = int((1 - validation_split) * len(x))

        train_x, train_y = x[:train_size], y[:train_size]
        # train_x = train_x.view(train_x.shape[0], 1, train_x.shape[1])
        # print(train_x.shape, train_y.shape)
        val_x, val_y = x[train_size:], y[train_size:]
        # train
        print("train_x shape", train_x.shape)
        print("train_y shape", train_y.shape)
        for epoch in range(epochs):
            # print(train_x.shape, train_y.shape)
            # outputs = self.forward(train_x)
            # print(outputs.shape)
            # loss = self.loss_fn(outputs, train_y)
            # loss.backward()
            # self.optimizer.step()
            # self.optimizer.zero_grad()
            # print(f'epoch {epoch}, loss {loss.item()}')

            for i in range(0, len(train_x), batch_size):
                batch_x, batch_y = train_x[i:i + batch_size], train_y[i:i + batch_size]
                # print(batch_x.shape, batch_y.shape)
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.forward(batch_x)
                loss = self.loss_fn(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                print(f'epoch {epoch}\tbatch {i}\tloss {loss.item()}')
            # validation
            with torch.no_grad():
                val_x, val_y = torch.tensor(val_x).to(self.device), torch.tensor(val_y).to(self.device)
                outputs = self.forward(val_x)

                loss = self.loss_fn(outputs, val_y)
                print(f'epoch {epoch}, validation loss {loss.item()}')
        # save model to file
        torch.save(self.state_dict(), self.get_name())
        return self

    def test(self, xy, x_in, headers):
        my_pos = (xy[:, -1, headers["tm-9-full"]])[:, :-1]
        x = torch.from_numpy(x_in).to(self.device)
        opp_pos_noise = self(x)[:, 8:10]
        opp_pos_noise = opp_pos_noise.cpu().detach().numpy() #TODO check .cpu
        opp_pos_noise[:, 0] *= config.max_x
        opp_pos_noise[:, 1] *= config.max_y
        opp_pos_full = (xy[:, -1, headers["opp-5-full"]])[:, :-1]

        my_dist = dist(my_pos, opp_pos_full)
        pos_count = (xy[:, -1, headers["opp-5-noise"]])[:, -1]
        error = dist(opp_pos_noise, opp_pos_full)

        all = np.zeros((error.shape[0], 3))
        all[:, 0] = error
        all[:, 1] = my_dist
        all[:, 2] = pos_count

        np.savetxt(f"{self.get_name('edp-')}", all, delimiter=',')


if __name__ == "__main__":
    model = TorchLSTM([10, 10, 10], ['relu', 'relu', 'relu'])
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()
    # model.compile(optimizer, loss_fn)
    print(model)
