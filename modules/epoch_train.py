"""
This module defines the epoch_train funtion which performs an epoch of training
of a given PyTorch neural network using the given dataloader, optimizer and criterion.
"""
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from .classifier import Classifier

def epoch_train(model: Classifier,
                train_loader: DataLoader,
                optimizer: Optimizer,
                criterion: nn.Module,
                epoch: int,
                max_epochs: int):
    
    """
    This function handles the training loop for a neural network model. It iterates over the dataset,
    computes the loss for each batch, performs backpropagation, and updates the model's weights.
    At the end of each epoch it prints the average loss.

    Params:
    model (Classifier): The neural network model of class Classifier to be trained.
    train_loader (DataLoader): DataLoader containing the training dataset.
                               It provides batches of data (images and labels) for training.
    optimizer (Optimizer): The optimization algorithm used for updating the model's parameters.
    criterion (nn.Module): The loss function used to measure the model's performance. In this architecture nn.CrossEntropyLoss() is recommended.
    epoch (int): The current epoch number during training.
    max_epochs (int): The total number of epochs for training.
    """
    # Set the model to training mode.
    model.train()

    # Initialize the epoch loss to 0.0.
    epoch_loss = 0.0
    
    # Iterate over the training data.
    for images, labels in train_loader:
        
        # Zero the gradients before forward pass.
        optimizer.zero_grad()

        # Forward pass: compute the model output for the batch.
        outputs = model(images)
        
        # Compute the loss between the model output and the ground truth labels.
        loss = criterion(outputs, labels)
        
        # Backward pass: compute the gradient of the loss w.r.t. model parameters.
        loss.backward()
        
        # Update the model parameters.
        optimizer.step()
        
        # Accumulate the  epoch loss.
        epoch_loss += loss.item()
    
    # Calculate the average loss over all batches in the train loader.
    av_loss = epoch_loss / len(train_loader)

    # Print the average loss for the current epoch.
    print(f"Epoch [{epoch + 1}/{max_epochs}], Loss: {av_loss:.4f}")
    
    return epoch_loss, av_loss
