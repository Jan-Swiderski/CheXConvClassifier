"""
This module defines a custom EarlyStopping class to implement early stopping feature in the
neural network training.
"""
from torch.optim import Optimizer
from .classifier import Classifier
from .create_checkpoint import create_checkpoint

class EarlyStopping:
    """
    A class for implementing early stopping during model training.
    """
    def __init__(self,
                 patience: int,
                 min_delta: float,
                 model: Classifier,
                 optimizer: Optimizer,
                 model_init_params: dict,
                 checkpoints_dir: str):
        
        """
        A class for implementing early stopping during model training.
        
        NOTE: When calling the class instance and passing the epoch number as an argument, you should not add 1 to it!
        The function create_checkpoint used in the class will handle
        the proper indexing with no need for user interference.

        Params:
            patience (int): The number of epochs with no improvement after which
                            training should be stopped.
            
            min_delta (float): The minimum change in the evaluation metric to
                            be considered as an improvement.
            
            model (Classifier): The neural network model as a instance of class Classifier being trained.
            
            optimizer (Optimizer): The optimizer used for training the model.
            
            model_init_params (dict): A dictionary containing parameters used to initialize the model of class classifier.
                                    When working with the Classifier class instace, these parameters are:
                                    l1_kernel_size (int): Kernel size of the first convolutional layer.
                                    l1_stride (int): Stride of the first convolutional layer.
                                    l1_out_chann (int): Number of output channels for the first convolutional layer.
                                    l2_kernel_size (int): Kernel size of the second convolutional layer.
                                    l2_stride (int): Stride of the second convolutional layer.
                                    l2_out_chann (int): Number of output channels for the second convolutional layer.
                                    l3_kernel_size (int): Kernel size of the third convolutional layer.
                                    l3_stride (int): Stride of the third convolutional layer.
                                    l3_out_chann (int): Number of output channels for the third convolutional layer.
                                    im_size (tuple): A tuple representing the input image size in the format (height, width).

            checkpoints_dir (str): The directory where checkpoints of the model
                                will be saved when improvements occur.

        Attributes:
            patience (int)
            min_delta (float)
            model (Classifier)
            optimizer (Optimizer)
            model_init_params (dict)
            checkpoints_dir (str)
            epoch (int)
            no_improve (int): Counter for consecutive epochs with no improvement.
            best_score (float): The best evaluation metric score achieved so far.

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
