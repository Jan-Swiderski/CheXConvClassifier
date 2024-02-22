"""
This module serves as the orchestrator for the entire neural network training process,
encompassing the initial setup, execution, and evaluation phases. It is designed to streamline
the workflow from loading and preprocessing data, through training and validation, to the final
evaluation on a test set. Key functionalities include environment setup, data loading and
transformation, model initialization, training loop execution with early stopping, and performance
evaluation.

Features:
- Load environment variables for paths to datasets and checkpoints.
- Create necessary directories for storing model checkpoints, specifically focusing on the best
  performing models based on validation accuracy.
- Initialize hyperparameters and model architecture parameters, allowing for easy adjustments to
  experiment with different configurations.
- Load datasets for training, validation, and testing phases, with support for dynamic image resizing
  and in-memory buffering to optimize performance.
- Define the model parameters, loss function, and optimizer, setting the stage for the training process.
- Utilize `torchsummary` for a comprehensive summary of the model, providing a detailed overview of the
  architecture, including layer types, output shapes, and parameter counts. This facilitates a deep
  understanding of the model's structure and complexity before initiating the training process, ensuring
  transparency and aiding in debugging and optimization efforts.
- Implement an early stopping mechanism to prevent overfitting and to ensure efficient training by
  halting the process once the model ceases to improve on the validation set.
- Conduct a thorough evaluation of the model on both validation and test datasets to gauge its
  performance and generalization capability.
- Visualize training and validation loss trends over epochs to monitor the learning process and
  diagnose potential issues.
"""
import time
import os
from dotenv import load_dotenv
import torch
from torch import nn
from torch import optim
from torchsummary import summary
from modules.get_datasets import get_datasets
from modules.get_dataloaders import get_dataloaders
from modules.classifier import Classifier
from modules.create_checkpoint import create_checkpoint
from modules.epoch_train import epoch_train
from modules.model_eval import model_eval
from modules.early_stopping import EarlyStopping
from modules.plot_losses import plot_losses

if __name__ == "__main__":

    start_time = time.time() # Start timing the execution of the script.

    # Load environment variables from .env file
    load_dotenv(dotenv_path = "./.env", override = True)
    CHEXPERT_ROOT = os.getenv('PATH_TO_CHEXPERT_ROOT')
    CHECKPOINTS_DIR = os.getenv('PATH_TO_CHECKPOINTS_DIR')

    # Create the directory for the best accuracy checkpoints in the main checkpoints directory
    BEST_ACC_CHECKPOINTS_DIR = os.path.join(CHECKPOINTS_DIR, "best_acc_checkpoints")

    try:
        os.makedirs(BEST_ACC_CHECKPOINTS_DIR)
        print(f"The directory {BEST_ACC_CHECKPOINTS_DIR} was created successfuly.")
    except:
        print(f"The directory {BEST_ACC_CHECKPOINTS_DIR} already exists.")
        
    # Define important filenames.
    TRAIN_DINFO_FILENAME = 'final_train_data_info.csv'
    TRAIN_IMAGES_DIRNAME = 'train_data'

    VALID_DINFO_FILENAME = 'final_valid_data_info.csv'
    VALID_IMAGES_DIRNAME = TRAIN_IMAGES_DIRNAME

    TEST_DINFO_FILENAME = "final_test_data_info.csv"
    TEST_IMAGES_DIRNAME = TRAIN_IMAGES_DIRNAME

    # Hyperparameters
    LEARNING_RATE = 0.001
    MOMENTUM = 0.9
    MAX_EPOCHS = 50
    TRAIN_BATCH_SIZE = 64
    VALID_BATCH_SIZE = 64
    TEST_BATCH_SIZE = 64
    MIN_MEM_AV_MB = 1024

    PATIENCE = 5 # Number of epochs with no improvement after which training will be stopped.
    MIN_IMPROVEMENT = 0.5 # Minimum improvement in validation accuracy to qualify as an improvement.

    IM_SIZE = (128, 128) # Input image size
    L1_OUT_CHANN = 8 # Number of filters in the first convolutional layer
    L1_KERNEL_SIZE = 5 # Size of the first layer's kernel
    L1_STRIDE = 1 # Stride of the first layer
    L2_OUT_CHANN = 16 # Number of filters in the second convolutional layer
    L2_KERNEL_SIZE = 3 # Size of the second layer's kernel
    L2_STRIDE = 1 # Stride of the second layer
    L3_OUT_CHANN = 32  # Number of channels in the third convolutional layer
    L3_KERNEL_SIZE = 5 # Size of the thrid layer's kernel
    L3_STRIDE = 1 # Stride of the third layer
    FC_OUT_FEATURES = 32 #Output features of the first fully connected layer
    
    # Create a dicttionary containing all of the network initialization parameters.
    # It will be further passed to the function creating checkpoints and saved in every checkpoint.
    net_init_params = {'l1_kernel_size': L1_KERNEL_SIZE,
                        'l1_stride': L1_STRIDE,
                        'l1_out_chann': L1_OUT_CHANN,
                        'l2_kernel_size': L2_KERNEL_SIZE,
                        'l2_stride': L2_STRIDE,
                        'l2_out_chann': L2_OUT_CHANN,
                        'l3_kernel_size': L3_KERNEL_SIZE,
                        'l3_stride': L3_STRIDE,
                        'l3_out_chann': L3_OUT_CHANN,
                        'fc_out_features': FC_OUT_FEATURES,
                        'im_size': IM_SIZE}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create datasets for training, validation, and testing.
    train_dataset, valid_dataset, test_dataset = get_datasets(chexpert_root = CHEXPERT_ROOT,
                                                              train_dinfo_filename = TRAIN_DINFO_FILENAME,
                                                              train_images_dirname = TRAIN_IMAGES_DIRNAME,
                                                              valid_dinfo_filename = VALID_DINFO_FILENAME,
                                                              vaild_images_dirname = VALID_IMAGES_DIRNAME,
                                                              test_dinfo_filename = TEST_DINFO_FILENAME,
                                                              test_images_dirname = TEST_IMAGES_DIRNAME,
                                                              ram_buffer_size_mb = MIN_MEM_AV_MB,
                                                              im_size = IM_SIZE)

    # Create data loaders
    train_loader, valid_loader, test_loader = get_dataloaders(train_dataset = train_dataset,
                                                              train_batch_size = TRAIN_BATCH_SIZE,
                                                              valid_dataset = valid_dataset,
                                                              valid_batch_size = VALID_BATCH_SIZE,
                                                              test_dataset = test_dataset,
                                                              test_batch_size = TEST_BATCH_SIZE)
    
    # Initialize the model with the specified architecture parameters.
    net = Classifier(**net_init_params)

    net.to(device)

    # Define the loss function (CrossEntropyLoss) and optimizer (SGD)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr = LEARNING_RATE, momentum = MOMENTUM)

    # Create an instance of a custom EarlyStopping class.
    earlystopper = EarlyStopping(patience = PATIENCE,
                                 min_delta = MIN_IMPROVEMENT,
                                 model = net,
                                 optimizer = optimizer,
                                 model_init_params = net_init_params,
                                 checkpoints_dir = BEST_ACC_CHECKPOINTS_DIR)
    
    
    print(f"Model initialization parameters: {net_init_params}", "/n")

    print(f"Model summary with the input size of: {IM_SIZE}", "/n")
    with torch.no_grad():
        summary(net, (1, 128, 128))


    # Training loop
    print("Starting training...")

    train_epoch_losses = []  # To store training loss after each epoch.
    val_total_losses = []  # To store validation loss after each epoch.
    train_av_losses = []  # To store average training loss.
    val_batch_losses = []  # To store batch-wise validation loss.

    for epoch in range(MAX_EPOCHS):
        # Print a new line for better readability between epochs in terminal output.
        print("\n")

        # Perform one epoch of training and return the losses.
        train_epoch_loss, train_av_loss = epoch_train(model = net,
                    train_loader = train_loader,
                    optimizer = optimizer,
                    criterion = criterion,
                    epoch = epoch,
                    max_epochs = MAX_EPOCHS)

        # Perform validation after each epoch and return the losses and accuracy.
        val_total_loss, val_batch_loss, val_accuracy = model_eval(model = net, 
                                                                criterion = criterion,
                                                                dataloader = valid_loader)

        print(f"Validiation accuracy at epoch {epoch + 1}: {val_accuracy:.4f}%")

        # Append the losses for plotting later.
        train_epoch_losses.append(train_epoch_loss)
        val_total_losses.append(val_total_loss)
        train_av_losses.append(train_av_loss)
        val_batch_losses.append(val_batch_loss)

        # Save a checkpoint after each epoch.
        create_checkpoint(model = net,
                        optimizer = optimizer,
                        model_init_params = net_init_params,
                        epoch = epoch,
                        checkpoints_dir = CHECKPOINTS_DIR,
                        accuracy = val_accuracy)
        
        # Check if early stopping criteria are met.
        if earlystopper(val_accuracy, epoch):
            break # Exit the training loop if early stopping criteria are met.
        
    print("Training finished")

    # Evaluate the model on the test dataset
    _, _, test_accuracy = model_eval(model = net,
                        dataloader = test_loader,
                        criterion = criterion)
    
    print("\n",f"Test accuracy: {test_accuracy:.4f}%", "\n")

    # Calculate and print the execution time of the whole script.
    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Execution time: {int(hours)}h {int(minutes)}m {round(seconds, 2)}s", "\n")
    
    # Plot the training and validation losses.
    plot_losses(train_epoch_losses = train_epoch_losses,
                val_total_losses = val_total_losses,
                train_av_losses = train_av_losses,
                val_batch_losses= val_batch_losses,
                checkpoints_dir = CHECKPOINTS_DIR)
