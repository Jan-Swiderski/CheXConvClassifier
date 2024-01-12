import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from modules import custom_CheXpert_Dataset
import os
from dotenv import load_dotenv

load_dotenv()
chexpert_root = os.getenv('PATH_TO_CHEXPERT_ROOT')
dinfo_filename = 'train_data_info.csv'
images_dirname = 'train_data'

chexpert = custom_CheXpert_Dataset.CheXpert(root_dir = chexpert_root,
                                            dinfo_filename = dinfo_filename,
                                            images_dirname = images_dirname,
                                            resize = True,
                                            custom_transforms = transforms.ToTensor()
                                            )

train_loader = DataLoader(dataset = chexpert,
                          batch_size = 64,
                          shuffle = True,
                          )


# Not finished yet
# class Classifier(nn.Module):
#     def __init__(self):
#         super(Classifier, self).__init__()
#         # Conv layer (3 x 3 kernel) with same padding
#         self.layer1conv = nn.Conv2d(in_channels = 1,
#                                     out_channels = 32
#                                     kernel_size = 3,
#                                     stride = 1,
#                                     padding = 2)
#         # Max pooling
#         maxpool = nn.MaxPool2d(kernel_size = 2,
#                                stride = 2)
        
#         # Conv layer (5 x 5 kernel) with same padding
#         self.layer2conv = nn.Conv2d(in_channels = 64,
#                                     out_channels = 128,
#                                     kernel_size= (5, 5),
#                                     stride = 1,
#                                     padding = 1)
        

    