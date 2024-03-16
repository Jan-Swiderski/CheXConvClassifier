"""
This module defines a custom convolutional neural network class called Classifier.
"""
import math
from torch import nn

class Classifier(nn.Module):
    """
    A custom convolutional neural network (CNN) classifier.
    This CNN consists of three convolutional layers followed by a ReLU activation function and a max pooling layer to reduce the spatial dimensions by half, and then two fully connected layers to produce the output predictions.
    NOTE: The input image is expected to have a single channel (e.g., grayscale).
    The network architecture is defined as follows:
    - Convolutional layer 1 -> ReLU -> Max Pooling
    - Convolutional layer 2 -> ReLU -> Max Pooling
    - Convolutional layer 3 -> ReLU -> Max Pooling
    - Fully Connected layer -> ReLU
    - Output Fully Connected layer
    """
    def __init__(self,
                 l1_kernel_size: int,
                 l1_stride: int,
                 l1_out_chann: int,
                 l2_kernel_size: int,
                 l2_stride: int,
                 l2_out_chann: int,
                 l3_kernel_size: int,
                 l3_stride: int,
                 l3_out_chann: int,
                 fc_out_features: int,
                 im_size: tuple[int, int]):
        """
    Initializes the classifier with specified configurations for each layer.
        
        Params:
            l1_kernel_size (int): Kernel size of the first convolutional layer.
            l1_stride (int): Stride of the first convolutional layer.
            l1_out_filters (int): Number of output filters for the first convolutional layer.
            l2_kernel_size (int): Kernel size of the second convolutional layer.
            l2_stride (int): Stride of the second convolutional layer.
            l2_out_filters (int): Number of output filters for the second convolutional layer.
            l3_kernel_size (int): Kernel size of the third convolutional layer.
            l3_stride (int): Stride of the third convolutional layer.
            l3_out_filters (int): Number of output filters for the third convolutional layer.
            fc_out_features (int): Number of neurons in the output layer of the fully connected layer.
            im_size (tuple[int, int]): The size of the input image (height, width).

        NOTE: The input image is expected to have a single channel (e.g., grayscale).

        """
        super(Classifier, self).__init__()
        self.im_height, self.im_width = im_size

        # Calculate padding for convolutional layers to achieve 'same' padding effect.
        self.l1_s_padding = self.calculate_same_padding(im_height = self.im_height,
                                                        im_width = self.im_width,
                                                        kernel_size = l1_kernel_size,
                                                        stride = l1_stride)

        # Initialize the number of input channels for the second convolutional layer.
        self.l2_in_chann = l1_out_chann

        # Calculate padding for the second convolutional layer.
        self.l2_s_padding = self.calculate_same_padding(im_height = self.im_height,
                                                        im_width = self.im_width,
                                                        kernel_size = l2_kernel_size,
                                                        stride = l2_stride)
        
        # Initialize the number of input channels for the third convolutional layer.
        self.l3_in_chann = l2_out_chann
        
        # Calculate padding for the third convolutional layer.
        self.l3_s_padding = self.calculate_same_padding(im_height = self.im_height,
                                                        im_width = self.im_width,
                                                        kernel_size = l3_kernel_size,
                                                        stride = l3_stride)

        # Calculate the number of input features for the fully connected layer.
        # This calculation accounts for the dimensionality reduction due to 3 max pooling layers, each reducing height and width by half.
        self.fc_in_features = int((self.im_height / (2**3)) * (self.im_width / (2**3)) * l3_out_chann)

        self.out_in_features = fc_out_features

        # Define the first convolutional layer using the given parameters.
        self.layer1conv = nn.Conv2d(in_channels = 1, # Assuming input images are grayscale
                                    out_channels = l1_out_chann,
                                    kernel_size = l1_kernel_size,
                                    stride = l1_stride,
                                    padding = self.l1_s_padding)
        
        # Define the second convolutional layer using the given parameters.
        self.layer2conv = nn.Conv2d(in_channels = self.l2_in_chann,  # Number of output channels of the 1st layer as the number input channels of the 2nd layer
                                    out_channels = l2_out_chann,
                                    kernel_size= l2_kernel_size,
                                    stride = l2_stride,
                                    padding = self.l2_s_padding)
        # Define the third convolutional layer using the given parameters.
        self.layer3conv = nn.Conv2d(in_channels = self.l3_in_chann, # Number of output channels of the 2nd layer as the number input channels of the 3rd layer.
                                    out_channels = l3_out_chann,
                                    kernel_size= l3_kernel_size,
                                    stride = l3_stride,
                                    padding = self.l3_s_padding)
        
        # Define the fully connected layer.
        self.fc = nn.Linear(in_features = self.fc_in_features, out_features = fc_out_features)

        self.out = nn.Linear(in_features = self.out_in_features, out_features = 3)

        # Define max pooling layer. Reduces the spatial size of the feature map by half.
        self.maxpool = nn.MaxPool2d(kernel_size = 2,
                               stride = 2)
        
        # Define ReLU activation
        self.relu = nn.ReLU()


    @staticmethod
    def calculate_same_padding(im_height:int,
                               im_width: int,
                               kernel_size: int,
                               stride: int):
        """
        Calculate padding to achieve 'same' padding effect for convolutional layers.

        This method calculates padding for convolutional layers to achieve the 'same' padding effect,
        ensuring that the output feature maps have the same spatial dimensions as the input.

        The formula used for calculating padding is:
        padding = max(ceil((stride - 1) * im_height - stride + kernel_size) / 2, 0)

        - 'stride' is the stride of the convolution operation.
        - 'kernel_size' is the size of the convolutional kernel.
        - 'im_height' is the height of the input feature map.
        - 'im_width' is the width of the input feature map.

        The 'ceil' function is used to round up the result of the calculation to the nearest integer.

        The 'max' function is used to ensure that the calculated padding is non-negative.

        Params:
            im_height (int): Height of the input feature map.
            im_width (int): Width of the input feature map.
            kernel_size (int): Size of the convolutional kernel.
            stride (int): Stride of the convolution operation.

        Returns:
            Tuple[int, int]: Padding values for height and width.
        """
        padding_height = max((math.ceil((stride - 1) * im_height - stride + kernel_size) / 2), 0)
        padding_width = max((math.ceil((stride - 1) * im_width - stride + kernel_size) / 2), 0)
        return int(padding_height), int(padding_width)
    

    def forward(self, x):
        """
        Defines the forward pass of the CNN classifier.

        Forward pass through the CNN. Applies consecutive convolutional layers with ReLU and max pooling,
        followed by flattening and passing through fully connected layers to produce output predictions.

        Params:
            x (Tensor): A tensor representing a batch of input images with shape [batch_size, channels, height, width].
            NOTE: The input image is expected to have a single channel (e.g., grayscale).

        Returns:
            Tensor: The output predictions of the network with shape [batch_size, num_classes].
        """
        # Forward pass through the first convolutional layer, followed by ReLU activation and max pooling.
        out = self.layer1conv(x)
        out = self.relu(out)
        out = self.maxpool(out)

        # Forward pass through the second convolutional layer, followed by ReLU activation and max pooling.

        out = self.layer2conv(out)
        out = self.relu(out)
        out = self.maxpool(out)
        # Forward pass through the third convolutional layer, followed by ReLU activation and max pooling.

        out = self.layer3conv(out)
        out = self.relu(out)
        out = self.maxpool(out)

        # Flatten the output tensor to prepare it for the fully connected layer, preserving the batch dimension.
        out = out.view(out.size(0), self.fc_in_features)

        # Forward pass through the first fully connected layer, followed by ReLU activation.        
        out = self.fc(out)
        out = self.relu(out)
        
        # Forward pass through the out fully connected layer which produces the final outcome.
        out = self.out(out)
        return out
