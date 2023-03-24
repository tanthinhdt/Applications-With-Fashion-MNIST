import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                     stride=stride, padding=1, bias=False))
        self.layers.append(nn.BatchNorm2d(out_channels))
        # self.relu = nn.ReLU(inplace=True)
        self.layers.append(nn.ReLU())

        self.layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                     stride=1, padding=1, bias=False))
        self.layers.append(nn.BatchNorm2d(out_channels))
        self.layers.append(nn.ReLU())

        self.downsample = downsample

    def forward(self, X):
        residual = X.clone()

        X = self.layers[0](X)
        X = self.layers[1](X)
        X = self.layers[2](X)

        X = self.layers[3](X)
        X = self.layers[4](X)

        if self.downsample is not None:
            residual = self.downsample(residual)
        X += residual

        X = self.relu(X)

        return X
    
    def get_block(self):
        return self.layers
