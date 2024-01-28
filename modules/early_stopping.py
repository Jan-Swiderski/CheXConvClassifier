from create_checkpoint import create_checkpoint
from torch.optim import Optimizer
from classifier import Classifier

class EarlyStopping:
    """
    A class for implementing early stopping during model training.
    
    NOTE: When calling the class instance and passing the epoch number as an argument, you should not add 1 to it!
    The function create_checkpoint used in the class will handle
    the proper indexing with no need for user interference.
    
    Early stopping is a technique used to prevent overfitting by monitoring
    a specified evaluation metric over consecutive epochs and stopping the
    training process when the metric does not improve for a certain number
    of consecutive epochs (patience).

    Params:
        patience (int): The number of epochs with no improvement after which
                        training should be stopped.
        min_delta (float): The minimum change in the evaluation metric to
                           be considered as an improvement.
        checkpoints_dir (str): The directory where checkpoints of the model
                               will be saved when improvements occur.
        model (Classifier): The neural network model as a instance of class Classifier being trained.
        optimizer (Optimizer): The optimizer used for training the model.

    Attributes:
        patience (int): The number of epochs with no improvement allowed.
        min_delta (float): The minimum change in the evaluation metric
                           to be considered as an improvement.
        checkpoints_dir (str): The directory where model checkpoints are saved.
        model (Classifier): The neural network model.
        optimizer (Optimizer): The optimizer used for training.
        epoch (int): The current epoch number.
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

            NOTE: When passing the epoch number as an argument, you should not add 1 to it!
            The function create_checkpoint used in the class will handle
            the proper indexing with no need for user interference.

        reset(self):
            Reset the counters and best score to start early stopping
            monitoring from scratch.

    Returns:
        bool: True if early stopping criteria are met, indicating that
              training should be stopped. False otherwise.
    """
    def __init__(self,
                 patience: int,
                 min_delta: float,
                 checkpoints_dir: str,
                 model: Classifier,
                 optimizer: Optimizer):
        # Initialize class attributes
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoints_dir = checkpoints_dir

        self.model = model
        self.optimizer = optimizer

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
                              checkpoints_dir = self.checkpoints_dir,
                              accuracy = new_score,
                              epoch = epoch)
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