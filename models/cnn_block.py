import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, channels, kernel_size, stride,
                 padding, activation='relu'):
        super(CNNBlock, self).__init__()
        self.layers = nn.ModuleList()

        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()

        for in_channel, out_channel in zip(channels, channels[1:]):
            self.layers.append(nn.Conv2d(in_channel, out_channel,
                                         kernel_size, stride, padding))
            self.layers.append(nn.BatchNorm2d(out_channel))
            self.layers.append(self.activation)

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)

    def get_block(self):
        return self.layers
