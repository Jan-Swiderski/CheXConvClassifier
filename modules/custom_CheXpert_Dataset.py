import torch
from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image

class CheXpert(Dataset):
    
    def __init__(self, root_dir:str, dinfo_filename:str, images_dirname:str, resize:bool = True, custom_size:tuple = (320, 320), custom_transforms = None):
        """
        A custom PyTorch Dataset for the CheXpert dataset.

        Args:
            root_dir (str): The root directory of the dataset.
            dinfo_filename (str): The filename of the data information CSV file.
            images_dirname (str): The directory name containing the images.
            resize (bool, optional): Whether to resize images. Default is True.
            custom_size (tuple, optional): A tuple representing the custom image size (height, width). Default is (320, 320).
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
        
        # Check if resizing is enabled and set the image transforms accordingly
        if resize:
            if custom_transforms:
                self.transforms = transforms.Compose([transforms.Resize((custom_size)), custom_transforms])
            else:
                self.transforms = transforms.Resize((custom_size))
        else:
            self.transforms = custom_transforms
        
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