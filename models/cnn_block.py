import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, channels, kernel_sizes, strides,
                 paddings, activations=None):
        super(CNNBlock, self).__init__()
        self.layers = nn.ModuleList()

        if not activations:
            activations = ['relu'] * len(kernel_sizes)

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
