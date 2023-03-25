import torch.nn as nn
from torch.nn import init
from models.cnn_block import CNNBlock
from models.dense_block import DenseBlock
from models.residual_block import ResidualBlock
from torchsummary import summary


class ResNet18(nn.Module):
    def __init__(self, img_size, img_channel, n_classes) -> None:
        super().__init__(ResNet18, self)
        self.img_size = img_size
        self.img_channel = img_channel
        self.features = nn.ModuleList()
        self.classifier = nn.ModuleList()

        # 1 convolutional layer
        # with kernel size 7x7, 64 filters, stride 2, padding 1, ReLU
        cnn_block1_channels = [self.img_channel, 64]
        cnn_block1_kernel_sizes = [7]
        cnn_block1_strides = [2]
        cnn_block1_paddings = [1]
        self.features.extend(CNNBlock(channels=cnn_block1_channels,
                                      kernel_sizes=cnn_block1_kernel_sizes,
                                      stride=cnn_block1_strides,
                                      padding=cnn_block1_paddings).get_block())
        img_size = (img_size - 7) // 2 + 1

        # 1 max pooling layer with kernel size 3x3, stride 2
        self.features.append(nn.MaxPool2d(kernel_size=3, stride=2))
        img_size = (img_size - 3) // 2 + 1

        # 1 residual block with kernel size 3x3, 64 filters, stride 1, padding 1
        res_block1_out_channels = 64
        self.features.extend(ResidualBlock(in_channels=cnn_block1_channels[-1],
                                           out_channels=res_block1_out_channels,
                                           stride=1).get_block())

        # 1 residual block with kernel size 3x3, 64 filters, stride 1, padding 1
        res_block2_out_channels = 64
        self.features.extend(ResidualBlock(in_channels=res_block1_out_channels,
                                           out_channels=res_block2_out_channels,
                                           stride=1).get_block())

        # 1 residual block with kernel size 3x3, 128 filters, stride 2, padding 1
        res_block3_out_channels = 128
        self.features.extend(ResidualBlock(in_channels=res_block2_out_channels,
                                           out_channels=res_block3_out_channels,
                                           stride=2).get_block())
        img_size = (img_size - 3) // 2 + 1

        # 1 residual block with kernel size 3x3, 128 filters, stride 1, padding 1
        res_block4_out_channels = 128
        self.features.extend(ResidualBlock(in_channels=res_block3_out_channels,
                                           out_channels=res_block4_out_channels,
                                           stride=1).get_block())

        # 1 residual block with kernel size 3x3, 256 filters, stride 2, padding 1
        res_block5_out_channels = 256
        self.features.extend(ResidualBlock(in_channels=res_block4_out_channels,
                                           out_channels=res_block5_out_channels,
                                           stride=2).get_block())
        img_size = (img_size - 3) // 2 + 1

        # 1 residual block with kernel size 3x3, 256 filters, stride 1, padding 1
        res_block6_out_channels = 256
        self.features.extend(ResidualBlock(in_channels=res_block5_out_channels,
                                           out_channels=res_block6_out_channels,
                                           stride=1).get_block())

        # 1 residual block with kernel size 3x3, 256 filters, stride 2, padding 1
        res_block7_out_channels = 512
        self.features.extend(ResidualBlock(in_channels=res_block6_out_channels,
                                           out_channels=res_block7_out_channels,
                                           stride=2).get_block())
        img_size = (img_size - 3) // 2 + 1

        # 1 residual block with kernel size 3x3, 256 filters, stride 1, padding 1
        res_block8_out_channels = 512
        self.features.extend(ResidualBlock(in_channels=res_block7_out_channels,
                                           out_channels=res_block8_out_channels,
                                           stride=1).get_block())

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
