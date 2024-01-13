# Project Progress

## 1. Moving CheXpert Files

All CheXpert files from "CheXpert-v1.0 batch x (train y)" folders have been consolidated into a single "train_data" folder within the CheXpert root directory.

## 2. CSV Modification Tool

I have developed the "csv_modifier.py" tool, which modifies the original CSV files (in this case, "train.csv") from the CheXpert dataset. The tool prepares the "Path" column for further use, concatenates the "Frontal/Lateral" and "AP/PA" columns, and drops unnecessary columns. At the end, it creates the new "train_data.csv" file.

## 3. custom_CheXpert_Dataset.py Module

I have created the "custom_CheXpert_Dataset.py" module, defining the CheXpert class. This class inherits from the PyTorch Dataset class and returns a tuple with a CheXpert image and its ground truth label. Specific transforms are applied to the image, and resizing is also supported. By default it is 320 x 320 pixels but any given picture dimensions are supported

## 4. Classifier Module (classifier.py)

The "classifier.py" module defines a custom convolutional neural network classifier. It includes three convolutional layers followed by a fully connected layer for image classification. The hyperparameters for these layers, as well as the input image size, are configurable. This module plays a crucial role in the project by providing the architecture for training and evaluating model on the CheXpert dataset.

## 5. main.py and Model Initialization

In the "main.py" file, I have implemented the main script for the project. It loads the CheXpert dataset using the "custom_CheXpert_Dataset.py" module and initializes the custom convolutional neural network classifier defined in the "classifier.py" module. The hyperparameters for the network and dataset are configured in this script, providing a solid foundation for model training.
