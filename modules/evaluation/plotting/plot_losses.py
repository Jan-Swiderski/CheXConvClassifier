"""
This module defines the plot_losses function which can be used to visualize the training and the validation losses dependencies
after training the PyTorch neural network.
"""
import os
import matplotlib.pyplot as plt

def plot_losses(train_epoch_losses: list[float],
                val_total_losses: list[float],
                train_av_losses: list[float],
                val_batch_losses:list[float],
                checkpoints_dir: str):
    """
    Plots the comparison of training and validation losses in two separate graphs.

    This function takes four lists of loss values and plots them in two separate graphs:
    1. Training Epoch Loss vs Validation Total Loss
    2. Training Average Loss vs Validation Batch Loss
    
    Each graph will have appropriate titles, axis labels, and legends for clear understanding.
    
    The function assumes that all input lists are of the same length and correspond to the same epochs/batches.
    If this is not the case, the graphs might be misleading.

    Params:
    - train_epoch_losses (list of float): A list containing the loss values for each epoch during training.
    - val_total_losses (list of float): A list containing the total loss values for each validation step.
    - train_av_losses (list of float): A list containing the average loss values for each training batch.
    - val_batch_losses (list of float): A list containing the loss values for each validation batch.
    - checkpoints_dir (str): The directory path where model checkpoints are stored. 
        The function will save the generated plots as a PNG file in this directory.
        This allows for easy tracking and comparison of model performance over time.
    """

    # Verify that all lists are of the same length
    assert len(train_epoch_losses) == len(val_total_losses) == len(train_av_losses) == len(val_batch_losses), \
        "All input lists must be of the same length"

    # Plotting Training Epoch Loss vs Validation Total Loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)  # First subplot in a 1x2 grid
    plt.plot(train_epoch_losses, label='Training Epoch Loss', color='blue', marker='o')
    plt.plot(val_total_losses, label='Validation Total Loss', color='red', marker='x')
    plt.title('Training Epoch Loss vs Validation Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plotting Training Average Loss vs Validation Batch Loss
    plt.subplot(1, 2, 2)  # Second subplot in a 1x2 grid
    plt.plot(train_av_losses, label='Training Average Loss', color='green', marker='o')
    plt.plot(val_batch_losses, label='Validation Batch Loss', color='orange', marker='x')
    plt.title('Training Average Loss vs Validation Batch Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Displaying the plots and creating the png file.
    plt.tight_layout()

    plots_png_dir = os.path.join(checkpoints_dir, "losses_plots.png")
    plt.savefig(plots_png_dir)
    print("Plots saved at:", plots_png_dir)
    
    plt.show()
