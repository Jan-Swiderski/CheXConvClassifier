"""
This module handles the testing pipeline for image classification models by evaluating
their performance on a designated test dataset, based on a model loaded from a checkpoint.
The primary function within, `model_test`, orchestrates this process, encompassing loading
the model, preparing the test dataset, evaluating performance metrics, and exporting images
of incorrectly classified instances for analysis.

This testing framework is designed to be invoked through a unified interface provided by
`main.py`, which acts as the centralized command-line entry point for both training and
testing workflows. The `main.py` script facilitates the parsing of command-line arguments
related to model checkpoints, dataset paths, hyperparameter settings, and other
configurations, streamlining the execution process for end-users.

Key modules utilized include PyTorch for deep learning operations, PIL for image processing,
and custom-defined modules for tasks such as model initialization, dataset preparation,
performance evaluation, and metric computation.

Usage Convention:
-----------------
The testing process is initiated by running `main.py` with arguments that specify the testing
mode, dataset path, model checkpoint, and other relevant configurations. This modular approach
ensures a consistent user experience and simplifies the process of switching between training
and testing phases, contributing to a maintainable and scalable codebase.

Example command to run testing through `main.py`:
python main.py test /path/to/dataset /path/to/checkpoint.pth --triplicate_channel

This example assumes the presence of a command-line interface setup in `main.py` for handling
the 'test' command and appropriately routing the specified arguments to the `model_test`
function in this module.
"""
import os
import json
from pathlib import Path
import sys
import subprocess
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchsummary import summary
from modules.nets.model_factory import model_factory
from modules.evaluation.model_eval import model_eval
from modules.dataset.custom_chexpert_dataset import CheXpert
from modules.evaluation.metrics import *

def model_test(dataset_root: str,
               checkpoint: str,
               hyperparams: str,
               filenames_config: str,
               triplicate_channel: bool,
               **kwargs):
    """
    Conducts the testing phase on a given dataset using a model loaded from a specified
    checkpoint. It calculates and prints test metrics including total loss and batch
    loss, and exports images of incorrectly predicted samples for further analysis.

    Parameters:
    - dataset_root (str): The root directory of the dataset.
    - checkpoint (str): Path to the model checkpoint file.
    - hyperparams (str): Path to the hyperparameters JSON file.
    - filenames_config (str): Path to the filenames configuration JSON file.
    - triplicate_channel (bool): Indicates whether to triplicate the channel for grayscale images.
    - **kwargs: Additional keyword arguments not currently used but included for future expandability.

    Returns:
    - None. Outputs include print statements of test metrics and exported images of incorrect predictions.
    """
    checkpoint_dict = torch.load(checkpoint)

    training_init_params = checkpoint_dict["training_init_params"]
    model_init_params = training_init_params["model_init_params"]

    training_init_params = checkpoint_dict["training_init_params"]
    model_init_params = training_init_params["model_init_params"]

    with open(hyperparams, "r") as hyperparams_json:
        hyperparams_dict = json.load(hyperparams_json)

    with open(filenames_config, "r") as filenames_config_json:
        filenames_dict = json.load(filenames_config_json)

    test_dinfo_filename, test_images_dirname = filenames_dict["TEST_DINFO_FILENAME"], filenames_dict["TEST_IMAGES_DIRNAME"]
    
    test_batch_size = hyperparams_dict["test_batch_size"]
    min_mem_av_mb = hyperparams_dict["min_mem_av_mb"]
    im_size = model_init_params["im_size"]

    model = model_factory(checkpoint=checkpoint_dict)

    if triplicate_channel:
        summary(model, (3, im_size[0], im_size[1]))
    else:
        summary(model, (1, im_size[0], im_size[1]))

    test_dataset_tensors = CheXpert(root_dir=dataset_root,
                                    dinfo_filename=test_dinfo_filename,
                                    images_dirname=test_images_dirname,
                                    ram_buffer_size_mb=min_mem_av_mb,
                                    custom_size=im_size,
                                    custom_transforms=transforms.ToTensor())
    
    test_dataset_pil = CheXpert(root_dir=dataset_root,
                                dinfo_filename=test_dinfo_filename,
                                images_dirname=test_images_dirname,
                                ram_buffer_size_mb=min_mem_av_mb,
                                custom_size=im_size)
    
    test_loader = DataLoader(dataset=test_dataset_tensors, batch_size=test_batch_size,
                             shuffle=False)
    
    criterion = nn.CrossEntropyLoss() 

    labels_dict = test_dataset_tensors.get_labels()

    confusion_matrix = ConfusionMatrix(labels_dict=labels_dict)
    accuracy_strategy = AccurracyStrategy(labels_dict=labels_dict)
    precision_strategy = PrecisionStrategy(labels_dict=labels_dict)
    recall_strategy = RecallStrategy(labels_dict=labels_dict)

    confusion_matrix.set_metric_strategy([accuracy_strategy, precision_strategy, recall_strategy])
    
    test_total_loss, test_batch_loss, _, incorrect_indexes = model_eval(model=model,
                                                                     criterion=criterion,
                                                                     dataloader=test_loader,
                                                                     confusion_matrix=confusion_matrix,
                                                                     triplicate_channel=triplicate_channel,
                                                                     get_incorrect_indexes=True)
    
    print("Test metrics:",
          f"Total loss: {test_total_loss}",
          f"Batch loss: {test_batch_loss}",
          sep="\n")
    confusion_matrix.print_metric()

    checkpoint_root = Path(checkpoint).parent

    incorrect_preds_export_path = os.path.join(checkpoint_root, "test_incorrect_preds_export")
    os.makedirs(incorrect_preds_export_path, exist_ok=True)

    from_pil_path = os.path.join(incorrect_preds_export_path, "from_pil_image")
    from_totensor_path = os.path.join(incorrect_preds_export_path, "from_totensor")

    os.makedirs(from_pil_path, exist_ok=True)
    os.makedirs(from_totensor_path, exist_ok=True)

    for idx in incorrect_indexes:
        pil_im_filename = f"pil_im_index_{str(idx).zfill(5)}.jpg"
        pil_im, _ = test_dataset_pil[idx]
        pil_im.save(os.path.join(from_pil_path, pil_im_filename))

        tens_im_filename = f"tens_im_idx{str(idx).zfill(5)}.jpg"
        tens_im, _ = test_dataset_tensors[idx]
        save_image(tensor=tens_im, fp=os.path.join(from_totensor_path, tens_im_filename))

    if sys.platform == "win32":
        subprocess.run(['explorer', incorrect_preds_export_path], check=True)
    elif sys.platform == "darwin":
        subprocess.run(['open', incorrect_preds_export_path], check=True)
    else:
        subprocess.run(['xdg-open', incorrect_preds_export_path], check=True)