import torch
from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image

class CheXpert(Dataset):
    
    def __init__(self, root_dir:str, dinfo_filename:str, images_dirname:str, resize:bool = True, custom_size:tuple = (320, 320), to_grayscale:bool = True, custom_transforms = None):
        """
        A custom PyTorch Dataset for the CheXpert dataset.

        Args:
            root_dir (str): The root directory of the dataset.
            dinfo_filename (str): The filename of the data information CSV file.
            images_dirname (str): The directory name containing the images.
            resize (bool, optional): Whether to resize images. Default is True.
            custom_size (tuple, optional): A tuple representing the custom image size (height, width). Default is (320, 320).
            to_grayscale(bool, optional): Whether to convert image to grayscale. Default is True.
            custom_transforms (callable, optional): A custom transform function to apply to images. Default is None.
        """
        # Load data information from the CSV file
        self.data_info = pd.read_csv(os.path.join(root_dir, dinfo_filename))
        self.root_dir = root_dir
        self.images_dirname = images_dirname

        # Define labels for each image type (One-hot encoding)
        self.labels = {'FrontalAP': torch.tensor([1, 0, 0]),
                       'FrontalPA': torch.tensor([0, 1, 0]),
                       'Lateral': torch.tensor([0, 0, 1])}

        
        # Dictionary defining transformation options based on input arguments
        # The keys represent combinations of resize, to_grayscale, and custom_transforms.
        # The values are corresponding transformation pipelines.
        self.trans_type = {
            # Case: Resize only:
            (True, False, False): transforms.Resize((custom_size)),
            # Case: Resize and convert to grayscale:
            (True, True, False): transforms.Compose([transforms.Resize((custom_size)), transforms.Grayscale(num_output_channels = 1)]),
            # Case: Resize and apply custom transforms:
            (True, False, True): transforms.Compose([transforms.Resize((custom_size)), custom_transforms]),
            # Case: Resize, convert to grayscale, and apply custom transforms:
            (True, True, True): transforms.Compose([transforms.Resize((custom_size)), transforms.Grayscale(num_output_channels = 1), custom_transforms]),
            # Case: No transformation:
            (False, False, False): None,
            # Case: Convert to grayscale and apply custom transforms:
            (False, True, True): transforms.Compose([transforms.Grayscale(num_output_channels = 1), custom_transforms]),
            # Case: Convert to grayscale only:
            (False, True, False): transforms.Grayscale(num_output_channels = 1),
            # Case: Apply custom transforms only:
            (False, False, True): custom_transforms,
        }

        # Determine the transformation pipeline based on input arguments
        # If any of resize, to_grayscale, or custom_transforms are True, select the appropriate pipeline.
        # Otherwise, set transforms to None for no transformation.
        self.transforms = self.trans_type[resize, to_grayscale, custom_transforms is not None]

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.data_info)
    
    def __getitem__(self, index):
        # Load an image and its corresponding label at the specified index
        image = Image.open(os.path.join(self.root_dir, self.images_dirname, self.data_info.loc[index, 'Path']))
        label = self.labels[self.data_info.loc[index, 'Frontal(AP/PA)/Lateral']]
    
    # Apply image transforms if they are defined
        if self.transforms:
            image = self.transforms(image)
        return (image, label)