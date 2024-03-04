"""
This module defines a custom EarlyStopping class to implement early stopping feature in the
neural network training.
"""
from torch import nn
from torch.optim import Optimizer
from .create_checkpoint import create_checkpoint

class EarlyStopping:
    """
    A class for implementing early stopping during model training.
    """
    def __init__(self,
                 patience: int,
                 min_delta: float,
                 model: nn.Module,
                 optimizer: Optimizer,
                 model_init_params: dict,
                 checkpoints_dir: str):
        
        """
        Initializes the EarlyStopping instance with specified parameters for monitoring
        and controlling the training process based on the evaluation metric.

        NOTE: When passing the epoch number as an argument, you should not add 1 to it!
        The function create_checkpoint used in the class will handle
        the proper indexing with no need for user interference.

        Params:
            patience (int): The number of epochs with no improvement after which
                            training should be stopped.
            
            min_delta (float): The minimum change in the evaluation metric to
                               be considered as an improvement.
            
            model (nn.Module): The neural network model being trained. This class is designed
                               to work with any model that is a subclass of torch.nn.Module.
            
            optimizer (Optimizer): The optimizer used for training the model.
            
            model_init_params (dict): A dictionary containing parameters used to initialize or train the model.
                                      This is necessary for creating checkpoints.
            
            checkpoints_dir (str): The directory where checkpoints of the model
                                   will be saved when improvements occur.

        Attributes:
            patience (int): Number of epochs to wait without improvement before stopping the training.
            min_delta (float): Minimum change in the monitored metric to qualify as an improvement.
            model (nn.Module): Model being trained.
            optimizer (Optimizer): Optimizer used for training.
            model_init_params (dict): Parameters used for model initialization.
            checkpoints_dir (str): Directory to save model checkpoints.
            no_improve (int): Counter for consecutive epochs without improvement.
            best_score (float): Best evaluation metric score achieved so far.


        Methods:
            __call__(self, new_score, epoch):
                Calls the _step function on a given arguments.

                NOTE: When passing the epoch number as an argument, you should not add 1 to it!
                The function create_checkpoint used in the class will handle
                the proper indexing with no need for user interference.

            _step(self, new_score, epoch):
                Check if the early stopping criteria are met based on a new evaluation score
                by examining whether the new evaluation score represents an improvement
                and updating the class attributes accordingly.

            reset(self):
                Reset the counters and best score to start early stopping
                monitoring from scratch.

        Returns:
            bool: True if early stopping criteria are met, indicating that
                training should be stopped. False otherwise.
        """
        # Initialize class attributes
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoints_dir = checkpoints_dir

        self.model = model
        self.optimizer = optimizer
        self.model_init_params = model_init_params

        # Initialize counters
        self.no_improve = 0
        self.best_score = 0

    def __call__(self, new_score: float, epoch: int):
        """
        Check if the early stopping criteria are met based on a new evaluation score
        by examining whether the new evaluation score represents an improvement
        and updating the class attributes accordingly.
        Params:
            new_score (float): A specified evaluation metric to be monitored.
            epoch (int): Current epoch number.
        
        Returns:
        bool: True if early stopping criteria are met, indicating that
              training should be stopped. False otherwise.
        NOTE: When passing the epoch number as an argument, you should not add 1 to it!
        The function create_checkpoint used in the class will handle
        the proper indexing with no need for user interference.
        
        """
        return self._step(new_score, epoch)
    
    def _step(self, new_score, epoch):
        if new_score >= self.best_score + self.min_delta:
            # Update best_score and reset no_improve counter
            self.best_score = new_score
            self.no_improve = 0
            # Create a checkpoint for the current model
            create_checkpoint(model = self.model,
                              optimizer = self.optimizer,
                              model_init_params = self.model_init_params,
                              epoch = epoch,
                              checkpoints_dir = self.checkpoints_dir,
                              accuracy = new_score)
            return False
        else:
            # Increment the no_improve counter
            self.no_improve += 1
        
        if self.no_improve >= self.patience:
            print("Early stopping triggered.")
            return True
    
    def reset(self):
        """
        Reset the counters and best score to start early stopping
        monitoring from scratch.
        """
        self.no_improve = 0
        self.best_score = 0
