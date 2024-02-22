"""
This module defines the model_eval funtion which can be further used to either test or validate the PyTorch
neural network model.
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
from .classifier import Classifier

def model_eval(model: Classifier,
               dataloader: DataLoader,
               criterion: nn.Module):

    """
    Evaluate a PyTorch neural network model on a given DataLoader.

    This function is used to validate the model on a validation dataset or test it on a test dataset.
    It calculates the model's accuracy and monitors the loss by predicting labels for the input data
    and comparing them to the ground truth.

    Params:
        model (Classifier): The neural network model of class Classifier to be validated.
        dataloader (DataLoader): DataLoader containing the dataset for evaluation.
                                 It provides batches of data (images and labels) for evaluation.
        criterion (nn.Module): The loss function used for calculating the loss.

    Returns:
            - total_loss (float): The total loss on the dataset.
            - av_loss (float): The average loss per batch.
            - accuracy (float): The accuracy of the model on the dataset, expressed as a percentage.
    """
    # Set the model to evaluation mode.
    model.eval()
    correct_preds = 0  # To count the number of correct predictions.
    total_preds = 0    # To count the total number of predictions made.
    total_loss = 0.0
    
    # Disable gradient calculations.
    with torch.no_grad():

        # Iterate over the validation data.
        for images, labels in dataloader:
            
            # Forward pass: compute the model output for the batch.
            outputs = model(images)

            # Calculate the loss for the batch.
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Get the index of the highest probability class for each image,
            # which represents the predicted label.
            _, predictions = torch.max(outputs, 1)

            # Update the total number of predictions made.
            total_preds += labels.size(0)

            # Update the number of correct predictions.
            correct_preds += (predictions == labels).sum().item()

    # Calculate the average loss per each batch.
    av_loss = total_loss / len(dataloader)

    # Calculate the accuracy as a percentage.
    accuracy = float(100 * correct_preds / total_preds)

    # Print the validation accuracy.
    # print(f'Validation accuracy: {accuracy:.2f}%')

    return total_loss, av_loss, accuracy
