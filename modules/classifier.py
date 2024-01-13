import torch
import torch.nn as nn


class Classifier(nn.Module):

    def __init__(self, l1_out_chann: int, l2_out_chann: int, l3_out_chann: int, im_size: tuple):
        """
        A custom convolutional neural network classifier.

        Args:
            l1_out_chann (int): Number of output channels for the first convolutional layer.
            l2_out_chann (int): Number of output channels for the second convolutional layer.
            l3_out_chann (int): Number of output channels for the third convolutional layer.
            im_size (tuple): A tuple representing the input image size in the format (height, width).
        """
        super(Classifier, self).__init__()
        self.l2_in_chann = l1_out_chann
        self.l3_in_chann = l2_out_chann
        self.im_height, self.im_width = im_size
        
        # Calculate the number of input features for the fully connected layer.
        # Dimensionality reduction due to 3 max pooling layers. Each reduces height and width by half.
        self.fc_in_features = int((self.im_height / (2**3)) * (self.im_width / (2**3)) * l3_out_chann)

        # Define the first convolutional layer (5 x 5 kernel) with same padding
        self.layer1conv = nn.Conv2d(in_channels = 1,
                                    out_channels = l1_out_chann,
                                    kernel_size = 5,
                                    stride = 1,
                                    padding = 2)
        
        # Define the second convolutional layer (3 x 3 kernel) with same padding
        self.layer2conv = nn.Conv2d(in_channels = self.l2_in_chann,
                                    out_channels = l2_out_chann,
                                    kernel_size= 3,
                                    stride = 1,
                                    padding = 1)
        # Define the third convolutional layer (3 x 3 kernel) with same padding
        self.layer3conv = nn.Conv2d(in_channels = self.l3_in_chann,
                                    out_channels = l3_out_chann,
                                    kernel_size= 3,
                                    stride = 1,
                                    padding = 1)
        
        # Define the fully connected layer
        self.fc = nn.Linear(in_features = self.fc_in_features, out_features = 3)

        # Define max pooling which reduces the height and width by half.
        self.maxpool = nn.MaxPool2d(kernel_size = 2,
                               stride = 2)
        
        # Define ReLU activation
        self.relu = nn.ReLU()

    def forward(self, x):
        # Forward pass through the first convolutional layer
        out = self.layer1conv(x)
        out = self.relu(out)
        out = self.maxpool(out)
        # Forward pass through the second convolutional layer
        out = self.layer2conv(out)
        out = self.relu(out)
        out = self.maxpool(out)
        # Forward pass through the third convolutional layer
        out = self.layer3conv(out)
        out = self.relu(out)
        out = self.maxpool(out)
        # Flatten the output tensor while preserving the batch dimension and prepare it for the forward pass through the fully connected layer
        out = out.view(out.size(0), self.fc_in_features)
        # Forward pass through the fully connected layer
        out = self.fc(out)
        return out