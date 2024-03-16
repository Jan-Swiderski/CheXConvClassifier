"""
This module orchestrates the process of fine-tuning a pre-trained neural network model for a new task, leveraging
transfer learning to improve efficiency and effectiveness. It encompasses the setup, execution, and evaluation phases
for adapting the MobileNet v3 model to classify images into three categories. The workflow includes loading and
preprocessing data, modifying the model architecture for the new task, training the model with early stopping, and
evaluating its performance on a test set. Key functionalities cover environment setup, dataset preparation, model
adaptation and initialization, training loop execution, and performance evaluation.

Features:
- Load environment variables for dataset and checkpoint paths.
- Ensure directory creation for storing model checkpoints, focusing on best validation accuracy.
- Initialize hyperparameters and adapt the MobileNet v3 architecture parameters for the classification task.
- Load datasets for training, validation, and testing phases, with support for dynamic image resizing
  and in-memory buffering to optimize performance.
- Modify the MobileNet v3 model by adjusting the first convolutional layer to accommodate input image channel size
  and replacing the final classification layer to fit the number of target classes.
- Utilize `torchsummary` for a comprehensive summary of the model, providing a detailed overview of the
  architecture, including layer types, output shapes, and parameter counts. This facilitates a deep
  understanding of the model's structure and complexity before initiating the training process, ensuring
  transparency and aiding in debugging and optimization efforts.
- Define loss function and optimizer tailored for fine-tuning the adapted model.
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
from torchvision import models
from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights
from torchsummary import summary
from modules.get_datasets import get_datasets
from modules.get_dataloaders import get_dataloaders
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
    CHECKPOINTS_ROOT = os.getenv('PATH_TO_CHECKPOINTS_DIR')

    CHECKPOINTS_DIR = os.path.join(CHECKPOINTS_ROOT, f"{time.strftime('%Y-%m-%d_%H-%M')}_mobile_net_v3_small")

    try:
        os.makedirs(CHECKPOINTS_DIR)
        print(f"The directory {CHECKPOINTS_DIR} was created successfuly.")
    except FileExistsError:
        print("Cannot initiate training in an existing directory. "
            "This directory may contain files from previous training sessions, risking data structure corruption. "
            "Please verify the environmental variables and all specified paths before attempting another run.")
    except Exception as e:
        print(f"An unexpected error occurred while creating the directory {CHECKPOINTS_DIR}: {str(e)}")

    # Create the directory for the best accuracy checkpoints in the main checkpoints directory
    BEST_ACC_CHECKPOINTS_DIR = os.path.join(CHECKPOINTS_DIR, "best_acc_checkpoints")
    try:
        os.makedirs(BEST_ACC_CHECKPOINTS_DIR)
        print(f"The directory {BEST_ACC_CHECKPOINTS_DIR} was created successfuly.")
    except Exception as e:
        print(f"An unexpected error occurred while creating the directory {BEST_ACC_CHECKPOINTS_DIR}: {str(e)}")
        
    # Define important filenames.
    TRAIN_DINFO_FILENAME = 'final_train_data_info.csv'
    TRAIN_IMAGES_DIRNAME = 'train_data'

    VALID_DINFO_FILENAME = 'final_valid_data_info.csv'
    VALID_IMAGES_DIRNAME = TRAIN_IMAGES_DIRNAME

    TEST_DINFO_FILENAME = "final_test_data_info.csv"
    TEST_IMAGES_DIRNAME = TRAIN_IMAGES_DIRNAME

    # Hyperparameters
    LEARNING_RATE = 0.0001
    MOMENTUM = 0.9
    MAX_EPOCHS = 50
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 32
    TEST_BATCH_SIZE = 32
    MIN_MEM_AV_MB = 1024

    PATIENCE = 5 # Number of epochs with no improvement after which training will be stopped.
    MIN_IMPROVEMENT = 0.0001 # Minimum improvement in validation accuracy to qualify as an improvement.

    IM_SIZE = (128, 128) # Input image size

    OPTIMIZER_TYPE = "torch.optim.Adam" # Just the information to save with checkpoints. Can come in handy if resuming training is needed.
    
    # Create a dicttionary containing all of the network initialization and training parameters. It also contains the optimizer.
    # It will be further passed to the function creating checkpoints and saved in every checkpoint.
    net_init_params = {'lr': LEARNING_RATE,
                       'momentum': MOMENTUM,
                       'optimizer_type': OPTIMIZER_TYPE,
                       'train_batch_size': TRAIN_BATCH_SIZE,
                       'valid_batch_size': VALID_BATCH_SIZE,
                       'test_batch_size': TEST_BATCH_SIZE,
                       'patience': PATIENCE,
                       'min_improvement': MIN_IMPROVEMENT,
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
    
    # Load the model with pretrained weights
    WEIGHTS = MobileNet_V3_Small_Weights.DEFAULT
    mobilenet_v3_small = models.mobilenet_v3_small(weights=WEIGHTS)

    # Retrieve the first convolutional layer
    first_conv_layer = mobilenet_v3_small.features[0][0]

    # Calculate the mean of the weights across the RGB channels of the first convolutional layer
    new_weights = first_conv_layer.weight.detach().mean(dim=1, keepdim=True)

    # Create a new first convolutional layer that accepts 1 input channel
    new_first_conv_layer = torch.nn.Conv2d(in_channels=1,
                                            out_channels=first_conv_layer.out_channels,
                                            kernel_size=first_conv_layer.kernel_size,
                                            stride=first_conv_layer.stride,
                                            padding=first_conv_layer.padding,
                                            bias=first_conv_layer.bias)

    # Assign the new weights to the new first layer
    new_first_conv_layer.weight.data = new_weights

    # Replace the original first layer with the new one
    mobilenet_v3_small.features[0][0] = new_first_conv_layer

    # Adjust the last layer for the new number of classes
    last_layer_in_features = mobilenet_v3_small.classifier[-1].in_features
    mobilenet_v3_small.classifier[-1] = torch.nn.Linear(last_layer_in_features, 3)

    mobilenet_v3_small.to(device)

    # Define the loss function (CrossEntropyLoss) and optimizer (Adam)
    criterion = nn.CrossEntropyLoss()

    # optimizer = optim.SGD(mobilenet_v3_small.parameters(), lr = LEARNING_RATE, momentum = MOMENTUM)
    optimizer = optim.Adam(params=mobilenet_v3_small.parameters(), lr=LEARNING_RATE)

    # Create an instance of a custom EarlyStopping class.
    earlystopper = EarlyStopping(patience = PATIENCE,
                                 min_delta = MIN_IMPROVEMENT,
                                 model = mobilenet_v3_small,
                                 optimizer = optimizer,
                                 trainig_init_params = net_init_params,
                                 checkpoints_dir = BEST_ACC_CHECKPOINTS_DIR)
    
    
    print(f"Model initialization and training parameters: {net_init_params}", "\n")

    print(f"Model summary with the input size of: {IM_SIZE}", "\n")
    with torch.no_grad():
        summary(mobilenet_v3_small, (1, IM_SIZE[0], IM_SIZE[1]))


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
        train_epoch_loss, train_av_loss = epoch_train(model = mobilenet_v3_small,
                    train_loader = train_loader,
                    optimizer = optimizer,
                    criterion = criterion,
                    epoch = epoch,
                    max_epochs = MAX_EPOCHS)

        # Perform validation after each epoch and return the losses and accuracy.
        val_total_loss, val_batch_loss, val_accuracy = model_eval(model = mobilenet_v3_small, 
                                                                criterion = criterion,
                                                                dataloader = valid_loader)

        print(f"Validiation accuracy at epoch {epoch + 1}: {val_accuracy:.4f}%")

        # Append the losses for plotting later.
        train_epoch_losses.append(train_epoch_loss)
        val_total_losses.append(val_total_loss)
        train_av_losses.append(train_av_loss)
        val_batch_losses.append(val_batch_loss)

        # Save a checkpoint after each epoch.
        create_checkpoint(model = mobilenet_v3_small,
                        optimizer = optimizer,
                        trainig_init_params = net_init_params,
                        epoch = epoch,
                        checkpoints_dir = CHECKPOINTS_DIR,
                        accuracy = val_accuracy)
        
        # Check if early stopping criteria are met.
        if earlystopper(val_accuracy, epoch):
            break # Exit the training loop if early stopping criteria are met.
        
    print("Training finished")

    # Evaluate the model on the test dataset
    _, _, test_accuracy = model_eval(model = mobilenet_v3_small,
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
