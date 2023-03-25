import torch.nn as nn


class DenseBlock(nn.Module):
    def __init__(self, dims, activations=['relu'], dropouts=[0.5]):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        n_layers = len(dims) - 1

        if len(activations) == 1:
            activations *= n_layers
        assert len(activations) == n_layers, "number of activation functions is invalid"

        if len(dropouts) == 1:
            dropouts *= n_layers
        assert len(dropouts) == n_layers, "number of dropouts is invalid"

        self.activation_dict = {
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
        }

        for i, (in_dim, out_dim) in enumerate(zip(dims, dims[1:])):
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.layers.append(nn.BatchNorm1d(out_dim))
            self.layers.append(self.activation_dict[activations[i]])
            self.layers.append(nn.Dropout(dropouts[i]))

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return X

    def get_block(self):
        return self.layers
