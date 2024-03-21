"""
This module provides the ConfusionMatrix class for managing and analyzing confusion matrices
in the context of classification problems. It is designed to support the incremental
update of the confusion matrix as new data becomes available and integrates with various
metric calculation strategies to evaluate model performance across multiple dimensions such
as precision, recall, and accuracy.

The design follows the strategy pattern for metric calculation, allowing for flexible
extension and customization of evaluation metrics without altering the core functionality
of the ConfusionMatrix class. This modular approach facilitates easy adaptation to different
evaluation needs and simplifies the integration of new metrics.
"""
import numpy as np
import torch
from .metric_strategies.metric_strategy import MetricStrategy

class ConfusionMatrix:
    """
    A class dedicated to the computation, storage, and manipulation of confusion matrices
    for classification tasks. It enables real-time updates to the matrix with new prediction
    outcomes and integrates with a flexible metric calculation framework based on the strategy
    pattern. This approach allows for the dynamic inclusion of different evaluation metrics
    through strategy objects.

    The convention used in this project utilizes the strategy pattern for metric calculation,
    enabling the ConfusionMatrix to work with different types of metrics dynamically. Each
    metric calculation strategy implements a common interface, allowing for consistent interaction
    and the possibility to extend functionality with new metric types easily.

    Attributes:
        __num_classes (int): Number of unique classes in the classification problem.
        __confmatrix (np.ndarray): 2D array representing the confusion matrix.
        __metric_strategy (MetricStrategy | list[MetricStrategy]): Strategy or list of strategies for metric calculation.
                                                                   "Metric strategy must be an instance of MetricStrategy
                                                                    or a list of MetricStrategy instances."

    Methods:
        __init__: Initializes a new instance of the ConfusionMatrix class.
        __call__: Returns the current state of the confusion matrix.
        step: Updates the confusion matrix based on new predictions and true labels.
        set_metric_strategy: Sets the metric calculation strategy.
        calculate_metric: Calculates metrics based on the current confusion matrix.
        print_metric: Prints the calculated metrics in a human-readable format.
    """
    def __init__(self,
                 labels_dict: dict,
                 ):
        """
        Initializes the ConfusionMatrix after deriving 
        number of classes from the dictionary of labels.
        
        Params:
            labels_dict (dict): A dictionary mapping class names to label indices.
        """
        # Determine the number of classes
        self.__num_classes = len(labels_dict)
        # Initialize the confusion matrix as a 2D numpy array of zeros
        self.__confmatrix = np.zeros(shape=(self.__num_classes, self.__num_classes))

        # Initially, no metric strategy is set. It can be later defined by the user
        self.__metric_strategy = None

    def __call__(self):
        """
        Returns the confusion matrix.
        
        Returns:
            np.ndarray: The current state of the confusion matrix.
        """
        return self.__confmatrix
    
    def step(self,
             ground_truth: torch.Tensor,
             quantized_preds: torch.Tensor):
        """
        Updates the confusion matrix with ground truth labels and quantized predictions.
        
        Params:
            ground_truth (torch.Tensor): The ground truth labels.
            quantized_preds (torch.Tensor): The predicted labels (quantized).
        """
        # Detach the tensors from the computation graph, move them to cpu
        # and convert them to numpy arrays
        ground_truth = ground_truth.detach().cpu().numpy()
        quantized_preds = quantized_preds.detach().cpu().numpy()

        # Update the confusion matrix based on the new batch of ground truth labels
        # and predictions. This operation counts the occurrences of each class-class pair
        np.add.at(self.__confmatrix, (ground_truth, quantized_preds), 1)

    def set_metric_strategy(self,
                            metric_strategy: MetricStrategy | list[MetricStrategy]):
        """
        Sets the metric calculation strategy or strategies.

        Params:
            metric_strategy (MetricStrategy | list[MetricStrategy]): A single metric strategy
            instance or a list of such instances to be used for metric calculation.
        """
        # Validate the input to ensure it's either a MetricStrategy instance or a list of such instances
        if isinstance(metric_strategy, list):
            if not all(isinstance(item, MetricStrategy) for item in metric_strategy):
                raise TypeError("All elements in the metric strategy list must be instances of MetricStrategy.")
            self.__metric_strategy = metric_strategy
            
        elif not isinstance(metric_strategy, MetricStrategy):
            raise TypeError("Metric strategy must be an instance of MetricStrategy or a list of MetricStrategy instances.")
        
        else:
            self.__metric_strategy = metric_strategy

    def calculate_metric(self,
                         as_percentage: bool = True,
                         as_dict: bool = True):
        """
        Calculates metrics based on the current state of the confusion matrix using the set
        metric strategy or strategies.

        Params:
            as_percentage (bool, optional): Whether to return the metric values as percentages.
                Defaults to True.
            as_dict (bool, optional): Whether to return the metrics in a dictionary format.
                Defaults to True.

        Returns:
            The calculated metric(s), format dependent on the `as_dict` and `as_percentage` arguments.
        """
        # Before calculating metrics, check if a metric strategy has been set
        if self.__metric_strategy is None:
            raise ValueError("Metric strategy has not been set")
        
        # Attempt to calculate the metric(s) using the current confusion matrix and the list of strategies 
        try:
            return [metric_strategy.calculate(self.__confmatrix, as_percentage, as_dict)
                    for metric_strategy in self.__metric_strategy]

        except:
            # Handle single metric strategy case
            return self.__metric_strategy.calculate(self.__confmatrix, as_percentage, as_dict)
        
    def print_metric(self,
                     as_percentage: bool = True,
                     decimal_points: int = 4):
        """
        Prints the calculated metrics in a human-readable format.

        Params:
            as_percentage (bool, optional): Whether the metric values are displayed as percentages.
                Defaults to True.
            decimal_points (int, optional): The number of decimal places for the metric values.
                Defaults to 4.
        """
        # Attempt to print the metric(s) using the current confusion matrix and the list of strategies 
        try:
            for metric_strategy in self.__metric_strategy:
                metric_strategy.print_metric(self.__confmatrix,
                                             as_percentage,
                                             decimal_points)
        except:
            # Handle single metric strategy case for printing
            self.__metric_strategy.print_metric(self.__confmatrix,
                                             as_percentage,
                                             decimal_points)
