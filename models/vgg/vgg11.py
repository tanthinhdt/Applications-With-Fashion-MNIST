import torch.nn as nn
from torch.nn import init
from models.cnn_block import CNNBlock
from models.dense_block import DenseBlock
from torchsummary import summary


class VGG11(nn.Module):
    def __init__(self, img_size, img_channel, n_classes):
        super(VGG11, self).__init__()
        self.img_size = img_size
        self.img_channel = img_channel
        self.features = nn.ModuleList()
        self.classifier = nn.ModuleList()

        # 1 convolutional layers with
        # 64 filters, kernel size 3x3, stride 1, padding 1, ReLU
        cnn_block1_channels = [self.img_channel, 64]
        cnn_block1_kernel_sizes = [3]
        cnn_block1_strides = [1]
        cnn_block1_paddings = [1]
        self.features.extend(CNNBlock(channels=cnn_block1_channels,
                                      kernel_sizes=cnn_block1_kernel_sizes,
                                      strides=cnn_block1_strides,
                                      paddings=cnn_block1_paddings).get_block())

        # max pooling layer with kernel size 2x2, stride 2
        self.features.append(nn.MaxPool2d(kernel_size=2, stride=2))
        img_size = (img_size - 2) // 2 + 1

        # 1 convolutional layers with
        # 128 filters, kernel size 3x3, stride 1, padding 1, ReLU
        cnn_block2_channels = [cnn_block1_channels[-1], 128]
        cnn_block2_kernel_sizes = [3]
        cnn_block2_strides = [1]
        cnn_block2_paddings = [1]
        self.features.extend(CNNBlock(channels=cnn_block2_channels,
                                      kernel_sizes=cnn_block2_kernel_sizes,
                                      strides=cnn_block2_strides,
                                      paddings=cnn_block2_paddings).get_block())

        # max pooling layer with kernel size 2x2, stride 2
        self.features.append(nn.MaxPool2d(kernel_size=2, stride=2))
        img_size = (img_size - 2) // 2 + 1

        # 2 convolutional layers with
        # 256 filters, kernel size 3x3, stride 1, padding 1, ReLU
        cnn_block3_channels = [cnn_block2_channels[-1], 256, 256]
        cnn_block3_kernel_sizes = [3]
        cnn_block3_strides = [1]
        cnn_block3_paddings = [1]
        self.features.extend(CNNBlock(channels=cnn_block3_channels,
                                      kernel_sizes=cnn_block3_kernel_sizes,
                                      strides=cnn_block3_strides,
                                      paddings=cnn_block3_paddings).get_block())
        # max pooling layer with kernel size 2x2, stride 2
        self.features.append(nn.MaxPool2d(kernel_size=2, stride=2))
        img_size = (img_size - 2) // 2 + 1

        # 2 convolutional layers with
        # 512 filters, kernel size 3x3, stride 1, padding 1, ReLU
        cnn_block4_channels = [cnn_block3_channels[-1], 512, 512]
        cnn_block4_kernel_sizes = [3]
        cnn_block4_strides = [1]
        cnn_block4_paddings = [1]
        self.features.extend(CNNBlock(channels=cnn_block4_channels,
                                      kernel_sizes=cnn_block4_kernel_sizes,
                                      strides=cnn_block4_strides,
                                      paddings=cnn_block4_paddings).get_block())

        # max pooling layer with kernel size 2x2, stride 2
        self.features.append(nn.MaxPool2d(kernel_size=2, stride=2))
        img_size = (img_size - 2) // 2 + 1

        # 2 convolutional layers with
        # 512 filters, kernel size 3x3, stride 1, padding 1, ReLU
        cnn_block5_channels = [cnn_block4_channels[-1], 512, 512]
        cnn_block5_kernel_sizes = [3]
        cnn_block5_strides = [1]
        cnn_block5_paddings = [1]
        self.features.extend(CNNBlock(channels=cnn_block5_channels,
                                      kernel_sizes=cnn_block5_kernel_sizes,
                                      strides=cnn_block5_strides,
                                      paddings=cnn_block5_paddings).get_block())

        # max pooling layer with kernel size 2x2, stride 2
        self.features.append(nn.MaxPool2d(kernel_size=2, stride=2))
        img_size = (img_size - 2) // 2 + 1

        # flatten layer
        self.features.append(nn.Flatten())

        # 2 fully connected layers with 4096 units, ReLu activation
        dense_block_dims = [img_size**2 * 512, 4096, 4096]
        dense_block_activations = ['relu']
        dense_block_dropouts = [0.5]
        self.classifier.extend(DenseBlock(dims=dense_block_dims,
                                          activations=dense_block_activations,
                                          dropouts=dense_block_dropouts).get_block())

        # softmax layer
        self.classifier.append(nn.Linear(dense_block_dims[-1], n_classes))
        self.classifier.append(nn.Softmax(dim=1))

        self.init_weights()

    def forward(self, X):
        for layer in self.features:
            X = layer(X)
        for layer in self.classifier:
            X = layer(X)
        return X

    def init_weights(self):
        for layer in self.features:
            if isinstance(layer, nn.Conv2d):
                init.kaiming_normal_(layer.weight)
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight)

    def get_summary(self):
        return summary(self, (self.img_channel, self.img_size, self.img_size))
