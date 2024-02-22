"""
This module defines the get_dataloaders function which creates and returns training, validation and test dataloaders
created using the corresponding datasets.
"""
from torch.utils.data import DataLoader
from .custom_chexpert_dataset import CheXpert

def get_dataloaders(train_dataset: CheXpert,
                    valid_dataset: CheXpert,
                    test_dataset: CheXpert,
                    train_batch_size: int = 64,
                    valid_batch_size: int = 64,
                    test_batch_size: int = 64,
                    train_shuffle: bool = True):
    """
    Function creates DataLoaders for the training, validation, and test datasets.

    Params:
    - train_dataset (CheXpert): Training dataset.
    - valid_dataset (CheXpert): Validation dataset.
    - test_dataset (CheXpert): Test dataset.
    - train_batch_size (int): Batch size for the training DataLoader. Default is 64.
    - valid_batch_size (int): Batch size for the validation DataLoader. Default is 64.
    - test_batch_size (int): Batch size for the test DataLoader. Default is 64.
    - train_shuffle (bool): Specifies whether the training data should be shuffled when creating the DataLoader. Default is True.

    Returns:
    - train_loader (DataLoader): DataLoader for the training dataset.
    - valid_loader (DataLoader): DataLoader for the validation dataset.
    - test_loader (DataLoader): DataLoader for the test dataset.
    """
    # Creating DataLoader for the training dataset
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=train_batch_size,
                              shuffle=train_shuffle)

    # Creating DataLoader for the validation dataset (no shuffling)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=valid_batch_size,
                              shuffle=False)

    # Creating DataLoader for the test dataset (no shuffling)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=test_batch_size,
                             shuffle=False)
    
    return train_loader, valid_loader, test_loader
