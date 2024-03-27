"""
This module provides functionality to load and prepare CheXpert datasets for model training, validation, and testing. 
It defines a `get_datasets` function that instantiates and returns datasets for each phase, utilizing the custom 
CheXpert dataset class for consistent data handling and transformations. 
"""
from typing import Optional
from torchvision import transforms
from .custom_chexpert_dataset import CheXpert

def get_datasets(chexpert_root: str,
                 ram_buffer_size_mb: int,
                 im_size: tuple[int, int],
                 custom_transforms: callable = transforms.ToTensor(),
                 train_dinfo_filename: Optional[str] = None,
                 train_images_dirname: Optional[str] = None,
                 valid_dinfo_filename: Optional[str] = None,
                 vaild_images_dirname: Optional[str] = None,
                 test_dinfo_filename: Optional[str] = None,
                 test_images_dirname: Optional[str] = None):
    """
    Instantiates and returns training, validation, and test datasets for the CheXpert medical image dataset.

    Parameters:
        chexpert_root (str): The root directory where the CheXpert dataset is stored.
        ram_buffer_size_mb (int): Size of the RAM buffer for loading images, in megabytes.
        im_size (tuple[int, int]): The target image size (height, width) for resizing images.
        custom_transforms (callable): Transformation function(s) to apply to each image. Defaults to `ToTensor()`.
        train_dinfo_filename (Optional[str]): Filename of CSV with training data info. None skips dataset creation.
        train_images_dirname (Optional[str]): Subdirectory with training images. None skips dataset creation.
        valid_dinfo_filename (Optional[str]): Filename of CSV with validation data info. None skips dataset creation.
        vaild_images_dirname (Optional[str]): Subdirectory with validation images. None skips dataset creation.
        test_dinfo_filename (Optional[str]): Filename of CSV with test data info. None skips dataset creation.
        test_images_dirname (Optional[str]): Subdirectory with test images. None skips dataset creation.

        Returns:
        Tuple[Optional[CheXpert], Optional[CheXpert], Optional[CheXpert]]:
        - train_dataset (CheXpert or None): Initialized training dataset if both `train_dinfo_filename` and 
          `train_images_dirname` are provided, else None.
        - valid_dataset (CheXpert or None): Initialized validation dataset if both `valid_dinfo_filename` and 
          `vaild_images_dirname` are provided, else None.
        - test_dataset (CheXpert or None): Initialized test dataset if both `test_dinfo_filename` and 
          `test_images_dirname` are provided, else None.
"""
    train_dataset = valid_dataset = test_dataset = None

    # Initialize the training dataset with provided parameters, converting images to grayscale and applying custom transformations
    if train_dinfo_filename and train_images_dirname:
        train_dataset = CheXpert(root_dir = chexpert_root,
                            dinfo_filename = train_dinfo_filename,
                            images_dirname = train_images_dirname,
                            ram_buffer_size_mb = ram_buffer_size_mb,
                            custom_size = im_size,
                            to_grayscale = True,
                            custom_transforms = custom_transforms)
        
    # Similar initialization for validation and test datasets.
    
    if valid_dinfo_filename and vaild_images_dirname:
        valid_dataset = CheXpert(root_dir = chexpert_root,
                                dinfo_filename = valid_dinfo_filename,
                                images_dirname = vaild_images_dirname,
                                ram_buffer_size_mb = ram_buffer_size_mb,
                                custom_size = im_size,
                                to_grayscale = True,
                                custom_transforms = custom_transforms)
        
    if test_dinfo_filename and test_images_dirname:
        test_dataset = CheXpert(root_dir = chexpert_root,
                                dinfo_filename = test_dinfo_filename,
                                images_dirname = test_images_dirname,
                                ram_buffer_size_mb = ram_buffer_size_mb,
                                custom_size = im_size,
                                to_grayscale = True,
                                custom_transforms = custom_transforms)
    
    return train_dataset, valid_dataset, test_dataset
