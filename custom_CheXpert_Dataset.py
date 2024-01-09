import torch
from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd
import os
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
chexpert_root = os.getenv('PATH_TO_CHEXPERT_ROOT')
dinfo_filename = 'train_data_info.csv'
images_dirname = 'train_data'

class CheXpert(Dataset):
    def __init__(self, root_dir, dinfo_filename, images_dirname, resize = True, custom_transforms = None):
        self.data_info = pd.read_csv(os.path.join(root_dir, dinfo_filename))
        self.root_dir = root_dir
        self.images_dirname = images_dirname
        self.labels = {'FrontalAP': torch.tensor([1, 0, 0]),
                       'FrontalPA': torch.tensor([0, 1, 0]),
                       'Lateral': torch.tensor([0, 0, 1])}
        if resize:
            if custom_transforms:
                self.transforms = transforms.Compose([transforms.Resize((320, 320)), custom_transforms])
            else:
                self.transforms = transforms.Resize((320, 320))
        else:
            self.transforms = custom_transforms
        
    def __len__(self):
        return len(self.data_info)
    
    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root_dir, self.images_dirname, self.data_info.loc[index, 'Path']))
        label = self.labels[self.data_info.loc[index, 'Frontal(AP/PA)/Lateral']]

        if self.transforms:
            image = self.transforms(image)
        return (image, label)