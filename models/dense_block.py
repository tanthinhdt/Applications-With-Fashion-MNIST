import torch.nn as nn


class DenseBlock(nn.Module):
    def __init__(self, dims, activations=None, dropouts=None):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()

        if not activations:
            activations = ['relu'] * (len(dims) - 1)

        if not dropouts:
            dropouts = [0.5] * (len(dims) - 1)

        activation_dict = {
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
        }

        for i, (in_dim, out_dim) in enumerate(zip(dims, dims[1:])):
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.layers.append(nn.BatchNorm1d(out_dim))
            self.layers.append(activation_dict[activations[i]])
            self.layers.append(nn.Dropout(dropouts[i]))

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return X

    def get_block(self):
        return self.layers
