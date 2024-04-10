"""
This module defines a custom create_checkpoint fuction which, as the name suggests, is meant to be used to save
the PyTorch neural network checkpoints.
"""
import os
import torch
from torch import nn
from torch.optim import Optimizer

def create_checkpoint(model: nn.Module,
                      optimizer: Optimizer,
                      trainig_init_params: dict,
                      epoch:int,
                      accuracy: float,
                      checkpoints_dir:str):
    """
    Create and save a checkpoint of a PyTorch model and its optimizer.

    This function takes a PyTorch model, its optimizer, initialization parameters dicitionary, current epoch number, accuracy,
    and a directory path where checkpoints will be saved. It constructs a dictionary
    containing the model state, optimizer state, and epoch number, then saves it as
    a checkpoint file with a filename based on accuracy and epoch number.

    Params:
        model (nn.Module): PyTorch model to be saved.
        optimizer (Optimizer): PyTorch optimizer that optimizes the model.
        trainig_init_params (dict): A dictionary containing all the information about the model type, optimizer type and their hyperparameters.
                                    Training hyperparameters are included too.
        epoch (int): Current epoch number.
        checkpoints_dir (str): Path to the directory where checkpoints will be saved.
        accuracy (float): Model accuracy to be included in the checkpoint filename.

        NOTE: When passing the epoch number as an argument, you should not add 1 to it! 
        The function will handle the proper indexing with no user interference.
        
    Raises:
        OSError: If there is an issue with creating the checkpoints directory.
        Exception: Handles other exceptions that may occur during checkpoint creation and saving.
    """
    # Format the accuracy as a string and replace '.' with '-' to avoid filename issues.
    accuracy_str = f"{accuracy:.2f}"
    accuracy_str = accuracy_str.replace('.', '-')

    # Create a dictionary containing the model state, optimizer state, and epoch number.
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch + 1,  # Adding 1 to the epoch number because PyTorch indexes epochs from 0.
        'training_init_params': trainig_init_params
    }

    try:
        os.makedirs(checkpoints_dir, exist_ok = True)
    except OSError as e:
        # Handle any OS-related exceptions during directory creation.
        print(f"An error occurred while creating the checkpoints directory: {str(e)}.")

    # Create the checkpoint filename based on accuracy and epoch number.
    checkpoint_filename = f"checkpoint_acc{accuracy_str}epoch{str(epoch + 1).zfill(3)}.pth"

    try:
        # Save the model and optimizer checkpoint.
        torch.save(checkpoint, os.path.join(checkpoints_dir, checkpoint_filename))
        print(f"Checkpoint saved as: {checkpoint_filename}")

    except Exception as e:
        # Handle any exceptions that may occur during checkpoint saving.
        print(f"An error has occurred while saving the checkpoint: {str(e)}.")
