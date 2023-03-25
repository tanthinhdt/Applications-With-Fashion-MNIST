import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, channels, kernel_sizes=[3], strides=[2],
                 paddings=[1], activations=['relu']):
        super(CNNBlock, self).__init__()
        self.layers = nn.ModuleList()
        n_layers = len(channels) - 1

        if len(kernel_sizes) == 1:
            kernel_sizes *= n_layers
        assert len(kernel_sizes) == n_layers, "number of kernel sizes is invalid"

        if len(strides) == 1:
            strides *= n_layers
        assert len(strides) == n_layers, "number of strides is invalid"

        if len(paddings) == 1:
            paddings *= n_layers
        assert len(paddings) == n_layers, "number of paddings is invalid"

        if len(activations) == 1:
            activations *= n_layers
        assert len(activations) == n_layers, "number of activation functions is invalid"

        self.activation_dict = {
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
        }

        for i, (in_channel, out_channel) in enumerate(zip(channels, channels[1:])):
            self.layers.append(nn.Conv2d(in_channel, out_channel,
                                         kernel_sizes[i], strides[i],
                                         paddings[i]))
            self.layers.append(nn.BatchNorm2d(out_channel))
            self.layers.append(self.activation_dict[activations[i]])

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)

    def get_block(self):
        return self.layers
