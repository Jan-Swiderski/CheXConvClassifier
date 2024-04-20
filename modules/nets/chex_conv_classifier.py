"""
This module defines CheXConvClassifier, a convolutional neural network for classifying chest X-ray images from the CheXpert dataset.
It features a customizable architecture with variable convolutional blocks and a residual learning approach to improve performance. 
This classifier is designed for integration with the `model_factory` framework.
"""

import torch
from torch import nn

class ResidualBlock(nn.Module):
    """
    A Residual Block for convolutional neural networks, used in the CheXConvClassifier.

    This block implements standard convolutional layers followed by batch normalization and ReLU activations. 
    It includes a skip connection to facilitate the learning of identity functions, enhancing training deep networks.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        """
        Initializes the ResidualBlock.

        Parameters:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (int, optional): Size of the convolving kernel. Defaults to 3.
            stride (int, optional): Stride of the convolution. Defaults to 1.
            padding (int, optional): Padding added to both sides of the input. Defaults to 1.
        """
        super(ResidualBlock, self).__init__()
        
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn_depthwise = nn.BatchNorm2d(in_channels)
        self.bn_pointwise = nn.BatchNorm2d(out_channels)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.depthwise(x)
        out = self.bn_depthwise(out)
        out = self.relu(out)
        out = self.pointwise(out)
        out = self.bn_pointwise(out)
        out = self.relu(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out
class CheXConvClassifier(nn.Module):
    """
    A custom convolutional classifier for the CheXpert dataset images.

    This classifier is structured to work with the `model_factory.py` module, which allows for dynamic instantiation and configuration of various model types.
    """
    def __init__(self, initial_filters: list[int] = [16], initial_kernel_sizes: list[int] = [7],
                 initial_strides: list[int] = [2], initial_paddings: list[int] = [3],
                 blocks_params: list[tuple[int, int]] = [(16, 32), (32, 64)],
                 num_classes: int = 3, **kwargs):
        """
        Initializes the CheXConvClassifier with customizable layers and configurations.

        Parameters:
            initial_filters (list of int): List of the number of filters for each initial convolutional layer.
            initial_kernel_sizes (list of int): List of kernel sizes for each initial convolutional layer.
            initial_strides (list of int): List of strides for each initial convolutional layer.
            initial_paddings (list of int): List of paddings for each initial convolutional layer.
            blocks_params (list of tuples): Parameters for the ResidualBlocks, where each tuple contains in_channels and out_channels.
            num_classes (int): Number of classes for the output layer.

        Note:
            - Ensure that the output channels of the last initial convolutional layer match the input channels of the first residual block.
            - Each subsequent residual block should have its input channels match the output channels of the previous block, 
              ensuring layer consistency and functional integrity of the model.
        """
        super(CheXConvClassifier, self).__init__()
        layers = []
        in_channels = 1
        
        for i in range(len(initial_filters)):
            layers.append(nn.Conv2d(in_channels, initial_filters[i], kernel_size=initial_kernel_sizes[i], stride=initial_strides[i], padding=initial_paddings[i]))
            layers.append(nn.ReLU())
            in_channels = initial_filters[i]

        for single_block_params in blocks_params:
            in_ch, out_ch = single_block_params
            layers.append(ResidualBlock(in_ch, out_ch))
            in_ch = out_ch

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        last_out_channels = blocks_params[-1][1]
        self.classifier = nn.Sequential(
            nn.Linear(last_out_channels, 128),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def get_model(**kwargs) -> CheXConvClassifier:
    return CheXConvClassifier(**kwargs)