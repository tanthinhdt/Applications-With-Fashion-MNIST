import torch.nn as nn
from torch.nn import init
from models.cnn_block import CNNBlock
from models.residual_block import ResidualBlock
from torchsummary import summary


class ResNet152(nn.Module):
    def __init__(self, img_size, img_channel, n_classes) -> None:
        super().__init__(ResNet152, self)
        self.img_size = img_size
        self.img_channel = img_channel
        self.features = nn.ModuleList()
        self.classifier = nn.ModuleList()

        # 1 convolutional layer with
        # kernel size 7x7, 64 filters, stride 2, padding 1, ReLU
        cnn_block1_channels = [self.img_channel, 64]
        cnn_block1_kernel_sizes = [7]
        cnn_block1_strides = [2]
        cnn_block1_paddings = [1]
        self.features.extend(CNNBlock(channels=cnn_block1_channels,
                                      kernel_sizes=cnn_block1_kernel_sizes,
                                      strides=cnn_block1_strides,
                                      paddings=cnn_block1_paddings).get_block())
        img_size = (img_size - 7) // 2 + 1

        # 1 max pooling layer with kernel size 3x3, stride 2
        self.features.append(nn.MaxPool2d(kernel_size=3, stride=2))
        img_size = (img_size - 3) // 2 + 1

        # 1 residual block with 2 convolutional layer, each has:
        # kernel size 3x3, 64 filters, stride 1, padding 1
        res_block1_channels = [cnn_block1_channels[-1], 64, 64]
        res_block1_kernel_sizes = [3]
        res_block1_strides = [1]
        res_block1_paddings = [1]
        self.features.extend(ResidualBlock(channels=res_block1_channels,
                                           kernel_sizes=res_block1_kernel_sizes,
                                           strides=res_block1_strides,
                                           paddings=res_block1_paddings).get_block())

        # 1 residual block with 2 convolutional layer, each has:
        # kernel size 3x3, 64 filters, stride 1, padding 1
        res_block2_channels = [res_block1_channels[-1], 64, 64]
        res_block2_kernel_sizes = [3]
        res_block2_strides = [1]
        res_block2_paddings = [1]
        self.features.extend(ResidualBlock(channels=res_block2_channels,
                                           kernel_sizes=res_block2_kernel_sizes,
                                           strides=res_block2_strides,
                                           paddings=res_block2_paddings).get_block())

        # 1 residual block contains:
        # 1 convolutional layer with kernel size 3x3, 128 filters, stride 2, padding 1
        # 1 convolutional layer with kernel size 3x3, 128 filters, stride 1, padding 1
        res_block3_channels = [res_block2_channels[-1], 128, 128]
        res_block3_kernel_sizes = [3]
        res_block3_strides = [2, 1]
        res_block3_paddings = [1]
        self.features.extend(ResidualBlock(channels=res_block3_channels,
                                           kernel_sizes=res_block3_kernel_sizes,
                                           strides=res_block3_strides,
                                           paddings=res_block3_paddings).get_block())
        img_size = (img_size - 3) // 2 + 1

        # 1 residual block with 2 convolutional layer, each has:
        # kernel size 3x3, 128 filters, stride 1, padding 1
        res_block4_channels = [res_block3_channels[-1], 128, 128]
        res_block4_kernel_sizes = [3]
        res_block4_strides = [1]
        res_block4_paddings = [1]
        self.features.extend(ResidualBlock(channels=res_block4_channels,
                                           kernel_sizes=res_block4_kernel_sizes,
                                           strides=res_block4_strides,
                                           paddings=res_block4_paddings).get_block())

        # 1 residual block contains:
        # 1 convolutional layer with kernel size 3x3, 256 filters, stride 2, padding 1
        # 1 convolutional layer with kernel size 3x3, 256 filters, stride 1, padding 1
        res_block5_channels = [res_block4_channels[-1], 256, 256]
        res_block5_kernel_sizes = [3]
        res_block5_strides = [2, 1]
        res_block5_paddings = [1]
        self.features.extend(ResidualBlock(channels=res_block5_channels,
                                           kernel_sizes=res_block5_kernel_sizes,
                                           strides=res_block5_strides,
                                           paddings=res_block5_paddings).get_block())
        img_size = (img_size - 3) // 2 + 1

        # 1 residual block with 2 convolutional layer, each has:
        # kernel size 3x3, 256 filters, stride 1, padding 1
        res_block6_channels = [res_block5_channels[-1], 256, 256]
        res_block6_kernel_sizes = [3]
        res_block6_strides = [1]
        res_block6_paddings = [1]
        self.features.extend(ResidualBlock(channels=res_block6_channels,
                                           kernel_sizes=res_block6_kernel_sizes,
                                           strides=res_block6_strides,
                                           paddings=res_block6_paddings).get_block())

        # 1 residual block contains:
        # 1 convolutional layer with kernel size 3x3, 512 filters, stride 2, padding 1
        # 1 convolutional layer with kernel size 3x3, 512 filters, stride 1, padding 1
        res_block7_channels = [res_block6_channels[-1], 512, 512]
        res_block7_kernel_sizes = [3]
        res_block7_strides = [2, 1]
        res_block7_paddings = [1]
        self.features.extend(ResidualBlock(channels=res_block7_channels,
                                           kernel_sizes=res_block7_kernel_sizes,
                                           strides=res_block7_strides,
                                           paddings=res_block7_paddings).get_block())
        img_size = (img_size - 3) // 2 + 1

        # 1 residual block with 2 convolutional layer, each has:
        # kernel size 3x3, 512 filters, stride 1, padding 1
        res_block8_channels = [res_block7_channels[-1], 256, 256]
        res_block8_kernel_sizes = [3]
        res_block8_strides = [1]
        res_block8_paddings = [1]
        self.features.extend(ResidualBlock(channels=res_block8_channels,
                                           kernel_sizes=res_block8_kernel_sizes,
                                           strides=res_block8_strides,
                                           paddings=res_block8_paddings).get_block())

        # 1 avarage pooling layer
        self.features.append(nn.AvgPool2d(kernel_size=7, stride=1))
        img_size = (img_size - 7) + 1

        # 1 softmax layer
        self.classifier.append(nn.Linear(img_size**2, n_classes))
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
