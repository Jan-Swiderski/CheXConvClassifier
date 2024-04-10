"""
This module provides functionality to create DataLoader instances as needed based on the datasets provided. It allows
for the selective initialization of training, validation, and test dataloaders, enhancing flexibility and resource
efficiency in various data handling scenarios.
"""
from typing import Optional
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

def get_dataloaders(train_dataset: Optional[Dataset] = None,
                    valid_dataset: Optional[Dataset] = None,
                    test_dataset: Optional[Dataset] = None,
                    train_batch_size: int = 64,
                    valid_batch_size: int = 64,
                    test_batch_size: int = 64,
                    train_shuffle: bool = True):
    """
    Creates DataLoader instances for datasets provided. Dataloaders for training, validation, or test datasets are
    initialized only if the corresponding datasets are provided. This allows for selective dataloader creation,
    enhancing flexibility and efficiency.
    
    Parameters:
    - train_dataset (Dataset, optional): Training dataset. Default is None.
    - valid_dataset (Dataset, optional): Validation dataset. Default is None.
    - test_dataset (Dataset, optional): Test dataset. Default is None.
    - train_batch_size (int): Batch size for the training DataLoader. Default is 64.
    - valid_batch_size (int): Batch size for the validation DataLoader. Default is 64.
    - test_batch_size (int): Batch size for the test DataLoader. Default is 64.
    - train_shuffle (bool): Specifies whether to shuffle training data. Default is True.

    Returns:
    Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    - train_loader (DataLoader or None): DataLoader for the training dataset if provided, else None.
    - valid_loader (DataLoader or None): DataLoader for the validation dataset if provided, else None.
    - test_loader (DataLoader or None): DataLoader for the test dataset if provided, else None.
    """
    train_loader = valid_loader = test_loader = None

    if train_dataset:
        # Initializing DataLoader for training dataset if provided
        train_loader = DataLoader(dataset=train_dataset,
                                batch_size=train_batch_size,
                                shuffle=train_shuffle)

    if valid_dataset:
       # Initializing DataLoader for validation dataset if provided (no shuffling)
        valid_loader = DataLoader(dataset=valid_dataset,
                                batch_size=valid_batch_size,
                                shuffle=False)

    if test_dataset:
        # Initializing DataLoader for test dataset if provided (no shuffling)
        test_loader = DataLoader(dataset=test_dataset,
                                batch_size=test_batch_size,
                                shuffle=False)
    
    return train_loader, valid_loader, test_loader
