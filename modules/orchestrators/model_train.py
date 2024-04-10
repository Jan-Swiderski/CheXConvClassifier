"""
This module contains the main training pipeline for different types of models
used for image classification tasks. It orchestrates the training process by
integrating various components such as dataset preparation, model initialization,
training, validation, early stopping, checkpointing, and loss visualization.

The module's core function, `model_train`, reads configuration files for model
parameters, optimizer settings, hyperparameters, and dataset information to set up
the training environment. It then conducts training and validation through multiple
epochs, implementing early stopping based on validation accuracy, and periodically
saves checkpoints.

Key external modules used include PyTorch for model definition and operations,
`torchsummary` for model summary visualization, and custom modules for dataset
handling, model and optimizer instantiation, training and evaluation routines,
checkpoint management, and plotting utilities.

Usage Convention:
-----------------
This module is designed to be used in conjunction with `main.py`, which serves as
the entry point for executing the training pipeline. `main.py` utilizes argparse to
parse command-line arguments specifying the model type, training mode, dataset
location, checkpoint paths, and configurations for the model, optimizer, and other
hyperparameters.

To initiate the training process, `main.py` should be executed from the command line
with appropriate arguments. It then calls the `model_train` function from this module,
passing the parsed arguments to configure and start the training session.

For example:
python main.py mobilenet_v3_small train /path/to/dataset /path/to/checkpoints_root \
              -m /path/to/model_config.json

This modular approach facilitates a clear separation between the command-line interface
and the training logic, promoting a structured and maintainable codebase.
"""
import json
import torch
from typing import Optional
from torch import nn
from torchsummary import summary
from modules.utils import create_checkpoints_subdir
from modules.dataset.get_datasets import get_datasets
from modules.dataset.get_dataloaders import get_dataloaders
from modules.nets.model_factory import model_factory
from modules.optimizer_factory import optimizer_factory
from modules.training.epoch_train import epoch_train
from modules.evaluation.model_eval import model_eval
from modules.training.create_checkpoint import create_checkpoint
from modules.evaluation.plotting.plot_losses import plot_losses
from modules.evaluation.metrics.confusion_matrix import ConfusionMatrix
from modules.evaluation.metrics.metric_strategies.accuracy_strategy import AccurracyStrategy
from modules.evaluation.metrics.metric_strategies.precision_strategy import PrecisionStrategy
from modules.evaluation.metrics.metric_strategies.recall_strategy import RecallStrategy
from modules.training.early_stopping import EarlyStopping

def model_train(model_type: str,
                dataset_root: str,
                checkpoint: str,
                optim_config: str,
                hyperparams: str,
                filenames_config: str,
                triplicate_channel: bool,
                model_config: Optional[str] = None,
                **kwargs):
    """
    Orchestrates the training pipeline for a specified model type.

    Params:
        model_type (str): The type of model to train.
        dataset_root (str): Root directory path of the dataset.
        checkpoins_root (str): Root directory path for saving model checkpoints.
        optim_config (str): Path to the optimizer configuration JSON file.
        hyperparams (str): Path to the hyperparameters JSON file.
        filenames_config (str): Path to the filenames configuration JSON file.
        model_config_ (Optional[str]): Path to the model configuration JSON file. Default is None.
                                                If no model configuration path is provided, the model will be intialized
                                                with the default parameters, broadly desribed in README.md

    This function integrates various stages of the training process: loading
    and preprocessing datasets, model initialization, optimizer setup, conducting
    training and validation epochs, implementing early stopping, checkpointing,
    and plotting training/validation loss graphs. It leverages external utility
    functions and classes for detailed operations within these stages.
    """
    # Load configuration files for model (if path provided), optimizer, hyperparameters, and dataset filenames.
    if model_config is not None:
        with open(model_config, 'r') as model_config_json:
            model_config_dict = json.load(model_config_json)
    else:
        # If path for model configuration is not provided,
        # initialize the configuration dictionary for proper model preparation.
        model_config_dict = {}

    with open(optim_config, 'r') as optim_config_json:
        optim_config_dict = json.load(optim_config_json)

    with open(hyperparams, 'r') as hyperparams_json:
        hyperparams_dict = json.load(hyperparams_json)

    with open(filenames_config, 'r') as filenames_config_json:
        filenames_dict = json.load(filenames_config_json)

    # Extract specific configurations and hyperparameters for easy access.
    train_dinfo_filename, train_images_dirname = filenames_dict['TRAIN_DINFO_FILENAME'], filenames_dict['TRAIN_IMAGES_DIRNAME']
    valid_dinfo_filename, valid_images_dirname = filenames_dict['VALID_DINFO_FILENAME'], filenames_dict['VALID_IMAGES_DIRNAME']
    learning_rate = hyperparams_dict['learning_rate']
    max_epochs = hyperparams_dict["max_epochs"]
    train_batch_size = hyperparams_dict["train_batch_size"]
    valid_batch_size = hyperparams_dict["valid_batch_size"]
    patience = hyperparams_dict["patience"]
    min_improvement = hyperparams_dict["min_improvement"]
    min_mem_av_mb = hyperparams_dict["min_mem_av_mb"]
    im_size = hyperparams_dict['im_size']

    # Update model configuration dictionary with dynamic parameters.
    model_config_dict['model_type'] = model_type
    model_config_dict['im_size'] = im_size

    optimizer_type_info, optimizer_init_params = optim_config_dict["optimizer_type_info"], optim_config_dict["optimizer_init_params"]
    optimizer_init_params['lr'] = learning_rate # Set learning rate in optimizer parameters.
    
    # Compile all initialization parameters for training for saving in checkpoint.
    training_init_params = {'training_hyperparams': hyperparams_dict,
                            'model_init_params': model_config_dict,
                            'optimizer_type_info': optimizer_type_info,
                            'optimizer_init_params': optimizer_init_params}
    
    # Create a subdirectory in checkpoints root directory for storing checkpoints.
    checkpoints_dir = create_checkpoints_subdir(checkpoints_root=checkpoint,
                              model_type=model_type)
    
    # Prepare datasets and dataloaders for training and validation.
    train_dataset, valid_dataset, _ = get_datasets(chexpert_root=dataset_root,
                                                   ram_buffer_size_mb=min_mem_av_mb,
                                                   im_size=im_size,
                                                   train_dinfo_filename=train_dinfo_filename,
                                                   train_images_dirname=train_images_dirname,
                                                   valid_dinfo_filename=valid_dinfo_filename,
                                                   vaild_images_dirname=valid_images_dirname)
    
    train_loader, valid_loader, _ = get_dataloaders(train_dataset=train_dataset,
                                                    valid_dataset=valid_dataset,
                                                    train_batch_size=train_batch_size,
                                                    valid_batch_size=valid_batch_size)
    
    # Initialize the model and optimizer based on configurations.
    model = model_factory(model_init_params=model_config_dict)

    optimizer = optimizer_factory(model_parameters=model.parameters(),
                                  optimizer_type_info=optimizer_type_info,
                                  optimizer_init_params=optimizer_init_params)
    
    criterion = nn.CrossEntropyLoss() # Define the loss function.

    earlystopper = EarlyStopping(patience=patience,
                                 min_delta=min_improvement)
    
    # Display model, optimizer, and training configurations.
    print(f"Model type: {model_type}",
    f"Model configuration: {model_config_dict}",
    f"Training hyperparameters: {hyperparams_dict}",
    f"Optimizer info: {optimizer_type_info}",
    f"Optimizer params: {optimizer_init_params}",
    sep="\n")

    print(f"Model summary with the input size of: {im_size}", "\n")
    
    with torch.no_grad(): # Prevent tracking history in autograd.
        # Display the model's architecture and parameter count for the specified input size.
        if triplicate_channel:
            summary(model, (3, im_size[0], im_size[1]))
        else:
            summary(model, (1, im_size[0], im_size[1]))

    # Get labels dict from the dataset
    labels_dict = valid_dataset.get_labels()

    # Instantiate the confusion matrix and the metric strategies
    confusion_matrix = ConfusionMatrix(labels_dict=labels_dict)
    accuracy_strategy = AccurracyStrategy(labels_dict=labels_dict)
    precision_strategy = PrecisionStrategy(labels_dict=labels_dict)
    recall_strategy = RecallStrategy(labels_dict=labels_dict)
    
    # Set the metric strategies for later calculation
    confusion_matrix.set_metric_strategy([accuracy_strategy, precision_strategy, recall_strategy])
    
    # Initialize lists to store loss metrics for visualization.
    train_epoch_losses = []  # To store training loss after each epoch.
    val_total_losses = []  # To store validation loss after each epoch.
    train_av_losses = []  # To store average training loss.
    val_batch_losses = []  # To store batch-wise validation loss.

    print("Starting training...")
    # Training loop: iterates over the number of specified epochs.
    for epoch in range(max_epochs):
        print("\n") # New line for readability between epochs in output.
    
        # Train for one epoch and return epoch and average batch losses.
        train_epoch_loss, train_av_loss = epoch_train(model = model,
                                                    train_loader = train_loader,
                                                    optimizer = optimizer,
                                                    criterion = criterion,
                                                    epoch = epoch,
                                                    max_epochs = max_epochs,
                                                    triplicate_channel=triplicate_channel)
        
        # Perform validation, returning total loss, batch losses, and accuracy.
        val_total_loss, val_batch_loss, val_metrics = model_eval(model = model, 
                                                                    criterion = criterion,
                                                                    dataloader = valid_loader,
                                                                    confusion_matrix=confusion_matrix,
                                                                    triplicate_channel=triplicate_channel)
        
        # Extract validation accuracy from the tuple
        val_accuracy, _, _ = val_metrics

        # Print validation metrics for current epoch
        print(f"Validation metrics for epoch {epoch+1}/{max_epochs}:")
        confusion_matrix.print_metric()

        # Append losses for future plotting.
        train_epoch_losses.append(train_epoch_loss)
        val_total_losses.append(val_total_loss)
        train_av_losses.append(train_av_loss)
        val_batch_losses.append(val_batch_loss)

        # Save checkpoint after each epoch.
        create_checkpoint(model=model,
                        optimizer=optimizer,
                        trainig_init_params=training_init_params,
                        epoch=epoch,
                        accuracy=val_accuracy,
                        checkpoints_dir=checkpoints_dir)
        
        # Check if early stopping criteria are met.                           
        if earlystopper(val_accuracy, epoch):
            break # Exit the training loop if early stopping criteria are met.
    
    print("Training finished")

    # Plot the training and validation losses.
    plot_losses(train_epoch_losses = train_epoch_losses,
                val_total_losses = val_total_losses,
                train_av_losses = train_av_losses,
                val_batch_losses= val_batch_losses,
                checkpoints_dir = checkpoints_dir)
    