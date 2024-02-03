import torch
import torch.nn as nn
from custom_CheXpert_Dataset import CheXpert
from torch.utils.data import random_split
import torchvision.transforms as transforms

def get_Datasets(chexpert_root: str,
                 train_dinfo_filename: str,
                 train_images_dirname: str,
                 valid_dinfo_filename: str,
                 vaild_images_dirname: str,
                 ram_buffer_size_mb: int,
                 im_size: tuple[int, int],
                 custom_transforms = transforms.ToTensor()):
    """
    Function get_Datasets creates training, validation, and test datasets for CheXpert.

    Params:
    - chexpert_root (str): Root directory of CheXpert dataset.
    - train_dinfo_filename (str): Name of the training dataset information file (csv).
    - train_images_dirname (str): Directory containing training images.
    - valid_dinfo_filename (str): Name of the training dataset information file (csv).
    - valid_images_dirname (str): Directory containing validation images.
    - im_size (Tuple[int, int]): Tuple specifying the desired image size (height, width).
    - custom_transforms (callable): A custom transform or list of transforms to apply to images. Has to be compatible with torchvision.transforms.
                                    Default is torchvision.transforms.ToTensor().

    Returns:
    - train_dataset (CheXpert): Training dataset as an instance of the CheXpert custom dataset.
    - valid_dataset (CheXpert): Validation dataset as an instance of the CheXpert custom dataset.
    - test_dataset (CheXpert): Test dataset as an instance of the CheXpert custom dataset.
    """

    # Initialize the training dataset
    
    train_dataset = CheXpert(root_dir = chexpert_root,
                        dinfo_filename = train_dinfo_filename,
                        images_dirname = train_images_dirname,
                        ram_buffer_size_mb = ram_buffer_size_mb,
                        custom_size = im_size,
                        to_grayscale = True,
                        custom_transforms = custom_transforms)
    
    # Define the test dataset size as the 30% of the train dataset.
    test_size = int(0.3 * len(train_dataset))

    # Redefine the train dateset size
    train_size = len(train_dataset) - test_size

    # Split the training dataset into training and test sets
    train_dataset, test_dataset = random_split(train_dataset, [train_size, test_size])

    # Initialize the validation dataset
    valid_dataset = CheXpert(root_dir = chexpert_root,
                            dinfo_filename = valid_dinfo_filename,
                            images_dirname = vaild_images_dirname,
                            ram_buffer_size_mb = ram_buffer_size_mb,
                            custom_size = im_size,
                            to_grayscale = True,
                            custom_transforms = custom_transforms)
                            
    return train_dataset, valid_dataset, test_dataset