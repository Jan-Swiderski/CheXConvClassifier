import json
import torch
from torch import nn
from torchvision import transforms
from torchsummary import summary
from modules.utils import create_checkpoints_subdir
from modules.dataset.get_datasets import get_datasets
from modules.dataset.get_dataloaders import get_dataloaders
from modules.model_factory import model_factory
from modules.optimizer_factory import optimizer_factory
from modules.training.epoch_train import epoch_train
from modules.evaluation.model_eval import model_eval
from modules.training.create_checkpoint import create_checkpoint
from modules.evaluation.plotting.plot_losses import plot_losses
from modules.training.early_stopping import EarlyStopping

def model_train(model_type: str,
                dataset_root: str,
                checkpoints_root: str,
                model_config_json_path: str,
                optim_config_json_path: str,
                hyperparams_json_path: str,
                filenames_config_json_path: str
                ):
    
    with open(model_config_json_path, 'r') as model_config_json:
        model_config_dict = json.load(model_config_json)
    
    with open(optim_config_json_path, 'r') as optim_config_json:
        optim_config_dict = json.load(optim_config_json)

    with open(hyperparams_json_path, 'r') as hyperparams_json:
        hyperparams_dict = json.load(hyperparams_json)

    with open(filenames_config_json_path, 'r') as filenames_config_json:
        filenames_dict = json.load(filenames_config_json)


    CHEXPERT_ROOT = dataset_root
    CHECKPOINTS_ROOT = checkpoints_root
        
    # Define important filenames.
    TRAIN_DINFO_FILENAME = filenames_dict['TRAIN_DINFO_FILENAME']
    TRAIN_IMAGES_DIRNAME = filenames_dict['TRAIN_IMAGES_DIRNAME']

    VALID_DINFO_FILENAME = filenames_dict['VALID_DINFO_FILENAME']
    VALID_IMAGES_DIRNAME = filenames_dict['VALID_IMAGES_DIRNAME']

    LEARNING_RATE = hyperparams_dict['learning_rate']
    MAX_EPOCHS = hyperparams_dict["max_epochs"]
    TRAIN_BATCH_SIZE = hyperparams_dict["train_batch_size"]
    VALID_BATCH_SIZE = hyperparams_dict["valid_batch_size"]
    PATIENCE = hyperparams_dict["patience"]
    MIN_IMPROVEMENT = hyperparams_dict["min_improvement"]
    MIN_MEM_AV_MB = hyperparams_dict["min_mem_av_mb"]
    IM_SIZE = hyperparams_dict['im_size']

    model_config_dict['model_type'] = model_type

    optimizer_type_info = optim_config_dict["optimizer_type_info"]
    optimizer_init_params = optim_config_dict["optimizer_init_params"]
    optimizer_init_params['lr'] = LEARNING_RATE
    
    training_init_params = {'training_hyperparams': hyperparams_dict,
                            'model_init_params': model_config_dict,
                            'optimizer_type_info': optimizer_type_info,
                            'optimizer_init_params': optimizer_init_params}
    
    checkpoints_dir = create_checkpoints_subdir(checkpoints_root=CHECKPOINTS_ROOT,
                              model_type=model_type)
    
    train_dataset, valid_dataset, _ = get_datasets(chexpert_root=CHEXPERT_ROOT,
                                                   ram_buffer_size_mb=MIN_MEM_AV_MB,
                                                   im_size=IM_SIZE,
                                                   train_dinfo_filename=TRAIN_DINFO_FILENAME,
                                                   train_images_dirname=TRAIN_IMAGES_DIRNAME,
                                                   valid_dinfo_filename=VALID_DINFO_FILENAME,
                                                   vaild_images_dirname=VALID_IMAGES_DIRNAME)
    
    train_loader, valid_loader, _ = get_dataloaders(train_dataset=train_dataset,
                                                    valid_dataset=valid_dataset,
                                                    train_batch_size=TRAIN_BATCH_SIZE,
                                                    valid_batch_size=VALID_BATCH_SIZE)
    
    model = model_factory(model_init_params=model_config_dict)

    optimizer = optimizer_factory(model_parameters=model.parameters(),
                                  optimizer_type_info=optimizer_type_info,
                                  optimizer_init_params=optimizer_init_params)
    
    criterion = nn.CrossEntropyLoss()

    earlystopper = EarlyStopping(patience=PATIENCE,
                                 min_delta=MIN_IMPROVEMENT)
    
    print(f"Model type: {model_type}",
    f"Model configuration: {model_config_dict}",
    f"Training hyperparameters: {hyperparams_dict}",
    f"Optimizer info: {optimizer_type_info}",
    f"Optimizer params: {optimizer_init_params}",
    sep="\n")

    print(f"Model summary with the input size of: {IM_SIZE}", "\n")
    with torch.no_grad():
        summary(model, (1, IM_SIZE[0], IM_SIZE[1]))

    print("Starting training...")

    train_epoch_losses = []  # To store training loss after each epoch.
    val_total_losses = []  # To store validation loss after each epoch.
    train_av_losses = []  # To store average training loss.
    val_batch_losses = []  # To store batch-wise validation loss.

    for epoch in range(MAX_EPOCHS):
        # Print a new line for better readability between epochs in terminal output.
        print("\n")
    
        train_epoch_loss, train_av_loss = epoch_train(model = model,
                                                    train_loader = train_loader,
                                                    optimizer = optimizer,
                                                    criterion = criterion,
                                                    epoch = epoch,
                                                    max_epochs = MAX_EPOCHS)
        
        # Perform validation after each epoch and return the losses and accuracy.
        val_total_loss, val_batch_loss, val_accuracy = model_eval(model = model, 
                                                                    criterion = criterion,
                                                                    dataloader = valid_loader)

        print(f"Validiation accuracy at epoch {epoch + 1}: {val_accuracy:.4f}%")

        # Append the losses for plotting later.
        train_epoch_losses.append(train_epoch_loss)
        val_total_losses.append(val_total_loss)
        train_av_losses.append(train_av_loss)
        val_batch_losses.append(val_batch_loss)

        # Save a checkpoint after each epoch.
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