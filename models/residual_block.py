import torch.nn as nn
from cnn_block import CNNBlock


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_sizes, strides, paddings, activations=None, downsample=None):
        super(ResidualBlock, self).__init__()
        self.layers = nn.ModuleList()

        self.activation_dict = {
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
        }

        self.layers.extend(CNNBlock(channels, kernel_sizes,
                           strides, paddings, activations).get_block())

        self.downsample = downsample

    def forward(self, X):
        residual = X.clone()
        if self.downsample:
            residual = self.downsample(residual)

        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                X += residual
            X = layer(X)

        return X

    def get_block(self):
        return self.layers
