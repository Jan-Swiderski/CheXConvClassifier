"""
Thins module defines the get_datasets function which creates and returns the training, validation and test dataset as
instances of a custom CheXpert dataset class.
"""
import torchvision.transforms as transforms
from .custom_chexpert_dataset import CheXpert

def get_datasets(chexpert_root: str,
                 train_dinfo_filename: str,
                 train_images_dirname: str,
                 valid_dinfo_filename: str,
                 vaild_images_dirname: str,
                 test_dinfo_filename: str,
                 test_images_dirname: str,
                 ram_buffer_size_mb: int,
                 im_size: tuple[int, int],
                 custom_transforms: callable = transforms.ToTensor()):
    """
    Function get_Datasets creates training, validation, and test datasets. 
    Each one is an instance of a custom CheXpert class.
    
    Params:
    - chexpert_root (str): Root directory of CheXpert dataset.
    - train_dinfo_filename (str): Name of the training dataset information file (csv).
    - train_images_dirname (str): Directory containing training images.
    - valid_dinfo_filename (str): Name of the validation dataset information file (csv).
    - valid_images_dirname (str): Directory containing validation images.
    - test_dinfo_filename (str): Name of the vtest dataset information file (csv).
    - test_images_dirname (str): Directory containing test images.
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
    

    # Initialize the validation dataset
    valid_dataset = CheXpert(root_dir = chexpert_root,
                            dinfo_filename = valid_dinfo_filename,
                            images_dirname = vaild_images_dirname,
                            ram_buffer_size_mb = ram_buffer_size_mb,
                            custom_size = im_size,
                            to_grayscale = True,
                            custom_transforms = custom_transforms)

    # Initialize the test dataset
    test_dataset = CheXpert(root_dir = chexpert_root,
                            dinfo_filename = test_dinfo_filename,
                            images_dirname = test_images_dirname,
                            ram_buffer_size_mb = ram_buffer_size_mb,
                            custom_size = im_size,
                            to_grayscale = True,
                            custom_transforms = custom_transforms)
    
    return train_dataset, valid_dataset, test_dataset
