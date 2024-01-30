import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from modules.custom_CheXpert_Dataset import CheXpert
from modules.get_Datasets import get_Datasets
from modules.get_DataLoaders import get_DataLoaders
from modules.classifier import Classifier
from modules.create_checkpoint import create_checkpoint
from modules.epoch_train import epoch_train
from modules.model_eval import model_eval
from modules.early_stopping import EarlyStopping
from modules.plot_losses import plot_losses
import os
from dotenv import load_dotenv

if __name__ == "__main__":

    # Load environment variables from .env file
    load_dotenv()
    chexpert_root = os.getenv('PATH_TO_CHEXPERT_ROOT')
    checkpoints_dir = os.getenv('PATH_TO_CHECKPOINTS_DIR')
    best_acc_checkpoints_dir = os.getenv('PATH_TO_BEST_ACC_CHECKPOINTS')

    # Define important filenames.
    train_dinfo_filename = 'train_data_info.csv'
    train_images_dirname = 'train_data'

    valid_dinfo_filename = 'valid_data_info.csv'
    vaild_images_dirname = 'valid_data'

    # Hyperparameters
    learning_rate = 0.001
    momentum = 0.9
    max_epochs = 10
    train_batch_size = 64
    valid_batch_size = 64
    test_batch_size = 64
    min_mem_av_mb = 1024

    patience = 3
    min_improvement = 0.005

    im_size = (320, 320) # Input image size
    l1_out_chann = 8 # Number of channels in the first convolutional layer
    l2_out_chann = 16 # Number of channels in the second convolutional layer
    l3_out_chann = 32  # Number of channels in the third convolutional layer
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create datasets
    train_dataset, valid_dataset, test_dataset = get_Datasets(chexpert_root = chexpert_root,
                                                              train_dinfo_filename = train_dinfo_filename,
                                                              train_images_dirname = train_images_dirname,
                                                              valid_dinfo_filename = valid_dinfo_filename,
                                                              vaild_images_dirname = vaild_images_dirname,
                                                              ram_buffer_size_mb = min_mem_av_mb
                                                              im_size = im_size)

    # Create data loaders
    train_loader, valid_loader, test_loader = get_DataLoaders(train_dataset = train_dataset,
                                                              valid_dataset = valid_dataset,
                                                              test_dataset = test_dataset)
    
    # Initialize the model
    net = Classifier(l1_out_chann = l1_out_chann,
                    l2_out_chann = l2_out_chann,
                    l3_out_chann = l3_out_chann,
                    im_size = im_size)

    net.to(device)

    # Define the loss function (CrossEntropyLoss) and optimizer (SGD)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr = learning_rate, momentum = momentum)

    earlystopper = EarlyStopping(patience = patience,
                                 min_delta = min_improvement,
                                 checkpoints_dir = best_acc_checkpoints_dir,
                                 model = net,
                                 optimizer = optimizer)

    # Training loop
    print("Starting training...")

    train_epoch_losses = []
    val_total_losses = []
    train_av_losses = []
    val_batch_losses = []

    for epoch in range(max_epochs):

        # Training
        train_epoch_loss, train_av_loss = epoch_train(model = net,
                    train_loader = train_loader,
                    optimizer = optimizer,
                    criterion = criterion,
                    epoch = epoch,
                    max_epochs = max_epochs)

        # Validation
        val_total_loss, val_batch_loss, val_accuracy = model_eval(model = net,
                                valid_loader = valid_loader)

        if earlystopper(val_accuracy, epoch):
            break
        
        train_epoch_losses.append(train_epoch_loss)
        val_total_losses.append(val_total_loss)
        train_av_losses.append(train_av_loss)
        val_batch_losses.append(val_batch_loss)

        # Save model checkpoint after each epoch.
        create_checkpoint(model = net,
                          optimizer = optimizer,
                          checkpoints_dir = checkpoints_dir,
                          accuracy = val_accuracy)
    


    print("Training finished")

    # Evaluate the model on the test dataset
    model_eval(model = net,
                test_loader = test_loader,
                criterion = criterion)
    
    plot_losses(train_epoch_losses = train_epoch_losses,
                val_total_losses = val_total_losses,
                train_av_losses = train_av_losses,
                val_batch_losses= val_batch_losses)