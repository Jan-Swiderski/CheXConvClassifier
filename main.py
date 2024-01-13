import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from modules.custom_CheXpert_Dataset import CheXpert
from modules.classifier import Classifier
import os
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()
chexpert_root = os.getenv('PATH_TO_CHEXPERT_ROOT')
dinfo_filename = 'train_data_info.csv'
images_dirname = 'train_data'

# Hyperparameters
im_size = (320, 320) # Input image size
l1_out_chann = 8 # Number of channels in the first convolutional layer
l2_out_chann = 16 # Number of channels in the second convolutional layer
l3_out_chann = 32  # Number of channels in the third convolutional layer

# Initialize the CheXpert dataset
chexpert = CheXpert(root_dir = chexpert_root,
                    dinfo_filename = dinfo_filename,
                    images_dirname = images_dirname,
                    resize = True,
                    custom_size = im_size,
                    custom_transforms = transforms.ToTensor())

# Initialize the DataLoader for the dataset
train_loader = DataLoader(dataset = chexpert,
                          batch_size = 2,
                          shuffle = True,
                          )

# Initialize the model
net = Classifier(l1_out_chann = l1_out_chann,
                 l2_out_chann = l2_out_chann,
                 l3_out_chann = l3_out_chann,
                 im_size = im_size)