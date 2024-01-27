import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from classifier import Classifier

def model_test(model: Classifier,
               test_loader: DataLoader):
    """
    This function evaluates the model's accuracy by predicting the labels of the test dataset
    and comparing them to the ground truth labels.
    Note: This function does not return a value; it only prints the test accuracy.

    Params:
    model (Classifier): The neural network model of class Classifier to be tested.
    test_loader (DataLoader): DataLoader containing the test dataset.
                              It provides batches of data (images and labels) for testing.
    
    """

    # Set the model to evaluation mode.
    model.eval()
    correct_preds = 0  # To count the number of correct predictions.
    total_preds = 0    # To count the total number of predictions made.

    # Disables gradient calculations.
    with torch.no_grad():

        # Iterate over the test data.
        for images, labels in test_loader:
            # Forward pass: compute the model output for the batch.
            outputs = model(images)

            # Get the index of the highest probability class for each image,
            # which represents the predicted label.
            _, predictions = torch.max(outputs, 1)

            # Update the total number of predictions made.
            total_preds += labels.size(0)

            # Update the number of correct predictions.
            correct_preds += (predictions == labels).sum().item()
        
    # Calculate the accuracy as a percentage.
    accuracy = 100 * correct_preds / total_preds

    # Print the test accuracy.
    print(f'Test accuracy: {accuracy:.2f}%')