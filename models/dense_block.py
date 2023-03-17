import torch.nn as nn


class DenseBlock(nn.Module):
    def __init__(self, dims, activation='relu', dropout=0.5):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()

        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()

        for in_dim, out_dim in zip(dims, dims[1:]):
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.layers.append(nn.BatchNorm1d(out_dim))
            self.layers.append(self.activation)
            self.layers.append(nn.Dropout(dropout))

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return X

    def get_block(self):
        return self.layers
