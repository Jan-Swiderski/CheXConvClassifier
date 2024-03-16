"""
This module defines the ConfusionMatrix class which offers methods to update the matrix with new predictions, 
compute key metrics such as precision, recall, and overall accuracy, and retrieve these metrics in various formats (e.g., as percentages). 
Additionally, it provides a user-friendly way to print these statistics, making it easier to evaluate and compare the performance of different models.

Key Features:
- Dynamically updates the confusion matrix with ground truth labels and predicted labels.
- Calculates and retrieves recall, precision, and accuracy metrics, both as raw values and percentages.
- Offers flexibility in data presentation through lists and dictionaries, catering to different analysis needs.
- Simplifies the process of model evaluation with clear and concise statistics printing.
"""
import numpy as np
import torch
from metrics.metric_strategy import MetricStrategy

class ConfusionMatrix:
    """
    A class for computing and storing the confusion matrix for a set of predictions and true labels.
    
    Attributes:
        __classes_dict (dict): A dictionary mapping label indices to class names.
        __num_classes (int): The number of unique classes.
        __confmatrix (np.ndarray): A 2D numpy array to hold the confusion matrix counts.
        
    Methods:
        __call__: Returns the confusion matrix.
        step: Updates the confusion matrix with a new set of predictions and true labels.
        get_recalls: Computes the recall for each class.
        get_precisions: Computes the precision for each class.
        get_accuracy: Computes the overall accuracy.
    """
    def __init__(self,
                 labels_dict: dict,
                 ):
        """
        Initializes the ConfusionMatrix object with a dictionary of labels.
        
        Params:
            labels_dict (dict): A dictionary mapping class names to label indices.
        """
        # Invert the labels_dict to map indices to names for easier access
        self.__classes_dict = {value: key for key, value in labels_dict.items()}
        # Determine the number of classes
        self.__num_classes = len(self.__classes_dict)
        # Initialize the confusion matrix as a 2D numpy array of zeros
        self.__confmatrix = np.zeros(shape=(self.__num_classes, self.__num_classes))

        self.__metric_strategy = None

    def __call__(self):
        """
        Returns the confusion matrix.
        
        Returns:
            np.ndarray: The current state of the confusion matrix.
        """
        return self.__confmatrix
    
    def step(self,
             gt_labels: torch.Tensor,
             quantized_preds: torch.Tensor):
        """
        Updates the confusion matrix with ground truth labels and quantized predictions.
        
        Args:
            gt_labels (torch.Tensor): The ground truth labels.
            quantized_preds (torch.Tensor): The predicted labels (quantized).
        """
        # Detach the tensors from the computation graph, move them to cpu
        # and convert them to numpy arrays
        gt_labels = gt_labels.detach().cpu().numpy()
        quantized_preds = quantized_preds.detach().cpu().numpy()

        # Update confusion matrix counts
        np.add.at(self.__confmatrix, (gt_labels, quantized_preds), 1)

    def set_metric_strategy(self,
                            metric_strategy: MetricStrategy | list[MetricStrategy]):
        self.__metric_strategy = metric_strategy

    def calculate_metric(self,
                         as_percentage: bool = False):
        if self.__metric_strategy is None:
            raise ValueError("Metric strategy has not been set")
        
        if isinstance(self.__metric_strategy, list):
            if not all(isinstance(item, MetricStrategy) for item in self.__metric_strategy):
                raise TypeError("All elements in the metric strategy list must be instances of MetricStrategy.")
        
            metrics = []
            for metric_strategy in self.__metric_strategy():
                metrics.append(metric_strategy.calculate(self.__confmatrix))
            return metrics

        elif isinstance(self.__metric_strategy, MetricStrategy):
            return self.__metric_strategy.calculate(self.__confmatrix)
        
        else:
            raise TypeError("Metric strategy must be an instance of MetricStrategy or a list of MetricStrategy instances.")
        