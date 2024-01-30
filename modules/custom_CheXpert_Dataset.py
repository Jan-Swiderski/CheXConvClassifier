import torch
from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os
import psutil

class CheXpert(Dataset):
    
    def __init__(self,
                 root_dir:str,
                 dinfo_filename:str,
                 images_dirname:str,
                 ram_buffer_size_mb: int,
                 custom_size:tuple = (320, 320),
                 to_grayscale:bool = True,
                 custom_transforms = None,
                 ):
        """
        A custom PyTorch Dataset for the CheXpert dataset.

        Params:
            root_dir (str): The root directory of the dataset.
            dinfo_filename (str): The filename of the data information CSV file.
            images_dirname (str): The directory name containing the images.
            custom_size (tuple, optional): A tuple representing the custom image size (height, width). If None, resizing is disabled. Default is (320, 320).
            to_grayscale(bool, optional): Whether to convert image to grayscale. Default is True.
            custom_transforms (optional): A custom transform or list of transforms to apply to images. Default is None.
        """
        # Load data information from the CSV file
        self.data_info = pd.read_csv(os.path.join(root_dir, dinfo_filename))
        self.root_dir = root_dir
        self.images_dirname = images_dirname
        self.ram_buffer_size_mb = ram_buffer_size_mb

        # Define labels for each image type
        self.labels = {'FrontalAP': 0,
                       'FrontalPA': 1,
                       'Lateral': 2}

        transforms_list = []

        if custom_size is not None:
            transforms_list.append(transforms.Resize(size = custom_size))
        
        if to_grayscale:
            transforms_list.append(transforms.Grayscale(num_output_channels = 1))

        if custom_transforms:
            transforms_list.extend(custom_transforms if isinstance(custom_transforms, list) else [custom_transforms])
        
        if transforms_list:
            self.transforms = transforms.Compose(transforms_list)
        else:
            self.transforms = None

    def is_memory_available(self):
        av_mem_mb = psutil.virtual_memory().available / (1024 ** 2)
        return av_mem_mb > self.ram_buffer_size_mb
        
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
