# Project Progress

## 1. Moving CheXpert Files

All CheXpert files from "CheXpert-v1.0 batch x (train y)" folders have been consolidated into a single "train_data" folder within the CheXpert root directory.
The folder "valid" from "CheXpert-v1.0 batch 1 (validate & csv)" has been moved to the CheXpert root directory and renamed to "valid_data"

## 2. CSV Modification Tool

I have developed the "csv_modifier.py" tool, which modifies the original CSV files (in this case, "train.csv" and "valid.csv") from the CheXpert dataset. The tool prepares the "Path" column for further use, concatenates the "Frontal/Lateral" and "AP/PA" columns, and drops unnecessary columns. At the end, it creates the new csv file with the desired name.

## 3. custom_CheXpert_Dataset.py Module

I have created the "custom_CheXpert_Dataset.py" module, defining the CheXpert class. This class inherits from the PyTorch Dataset class and returns a tuple with a CheXpert image and its ground truth label (One-got encoding). Custom transforms can be applied to the image, and both resizing (to 320 x 320 pixels) and converting the image to grayscale are supported.

## 4. Classifier Module (classifier.py)

The "classifier.py" module defines a custom convolutional neural network classifier. It includes three convolutional layers followed by a fully connected layer for image classification. The hyperparameters for these layers, as well as the input image size, are configurable. This module plays a crucial role in the project by providing the architecture for training and evaluating model on the CheXpert dataset.

## 5. Model Training and Evaluation in main.py

In the "main.py" file, I have implemented the main script for the project, where model training and evaluation take place. This script leverages the CheXpert dataset loaded using the "custom_CheXpert_Dataset.py" module and initializes the custom convolutional neural network classifier defined in the "classifier.py" module. The hyperparameters for the network and dataset are configured in this script, providing a solid foundation for model training and evaluation. The key functionalities added to "main.py" include:

- Loading and preprocessing the CheXpert dataset.
- Defining the neural network architecture using the custom classifier.
- Configuring hyperparameters such as learning rate, momentum, and batch size.
- Training the model using a training loop.
- Evaluating the model on validation and test datasets.

This script serves as the central component of the project, orchestrating the entire machine learning pipeline.

