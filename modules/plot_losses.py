import matplotlib.pyplot as plt

def plot_losses(train_epoch_losses: list[float],
                val_total_losses: list[float],
                train_av_losses: list[float],
                val_batch_losses:list[float]):
    """
    Plots the comparison of training and validation losses in two separate graphs.

    This function takes four lists of loss values and plots them in two separate graphs:
    1. Training Epoch Loss vs Validation Total Loss
    2. Training Average Loss vs Validation Batch Loss

    Params:
    - train_epoch_losses (list of float): A list containing the loss values for each epoch during training.
    - val_total_losses (list of float): A list containing the total loss values for each validation step.
    - train_av_losses (list of float): A list containing the average loss values for each training batch.
    - val_batch_losses (list of float): A list containing the loss values for each validation batch.

    The function doesn't return anything but plots the graphs using matplotlib.

    Each graph will have appropriate titles, axis labels, and legends for clear understanding.
    
    The function assumes that all input lists are of the same length and correspond to the same epochs/batches.
    If this is not the case, the graphs might be misleading.
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
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Displaying the plots
    plt.tight_layout()
    plt.show()