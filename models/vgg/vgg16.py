import torch.nn as nn
from torch.nn import init
from models.cnn_block import CNNBlock
from models.dense_block import DenseBlock
from torchsummary import summary


class VGG16(nn.Module):
    def __init__(self, img_size, img_channel, n_classes):
        super(VGG16, self).__init__()
        self.img_size = img_size
        self.img_channel = img_channel
        self.n_classes = n_classes
        self.features = nn.ModuleList()
        self.classifier = nn.ModuleList()

        # first 2 convolutional layers with
        # 64 filters, kernel size 3x3, stride 1, padding 1, ReLU
        cnn_block1_channels = [self.img_channel, 64, 64]
        self.features.extend(CNNBlock(channels=cnn_block1_channels,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1).get_block())

        # 1 max pooling layer with kernel size 2x2, stride 2
        self.features.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.img_size = (self.img_size - 2) // 2 + 1

        # 2 convolutional layers with
        # 128 filters, kernel size 3x3, stride 1, padding 1, ReLU
        cnn_block2_channels = [cnn_block1_channels[-1], 128, 128]
        self.features.extend(CNNBlock(channels=cnn_block2_channels,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1).get_block())

        # 1 max pooling layer with kernel size 2x2, stride 2
        self.features.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.img_size = (self.img_size - 2) // 2 + 1

        # 3 convolutional layers with
        # 256 filters, kernel size 3x3, stride 1, padding 1, ReLU
        cnn_block3_channels = [cnn_block2_channels[-1], 256, 256, 256]
        self.features.extend(CNNBlock(channels=cnn_block3_channels,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1).get_block())

        # 1 max pooling layer with kernel size 2x2, stride 2
        self.features.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.img_size = (self.img_size - 2) // 2 + 1

        # 3 convolutional layers with
        # 512 filters, kernel size 3x3, stride 1, padding 1, ReLU
        cnn_block4_channels = [cnn_block3_channels[-1], 512, 512, 512]
        self.features.extend(CNNBlock(channels=cnn_block4_channels,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1).get_block())

        # 1 max pooling layer with kernel size 2x2, stride 2
        self.features.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.img_size = (self.img_size - 2) // 2 + 1

        # 3 convolutional layers with
        # 512 filters, kernel size 3x3, stride 1, padding 1, ReLU
        cnn_block5_channels = [cnn_block4_channels[-1], 512, 512, 512]
        self.features.extend(CNNBlock(channels=cnn_block5_channels,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1).get_block())

        # 1 max pooling layer with kernel size 2x2, stride 2
        self.features.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.img_size = (self.img_size - 2) // 2 + 1

        # 1 flatten layer
        self.features.append(nn.Flatten())

        # 2 fully connected layer with 4096 units, ReLu activation
        dense_block_dims = [self.img_size**2 * 512, 4096, 4096]
        self.classifier.extend(DenseBlock(dims=dense_block_dims,
                                          activation='relu',
                                          dropout=0.5).get_block())

        # 1 softmax layer
        self.classifier.append(nn.Linear(dense_block_dims[-1], self.n_classes))

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
        model = nn.ModuleList()
        model.extend(self.features)
        model.extend(self.classifier)
        return summary(model, (self.img_channel, self.img_size, self.img_size))
