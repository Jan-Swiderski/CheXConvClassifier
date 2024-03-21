"""
This module introduces the abstract base class MetricStrategy, designed to serve as a 
blueprint for implementing various metric calculation strategies for evaluating classification 
models. Each derived class specifies how a particular metric (e.g., accuracy, precision, recall) 
is calculated using a confusion matrix. The module supports the strategy design pattern, 
allowing for flexible extension and integration of new metrics without altering the core 
evaluation mechanisms.
"""
from abc import ABC, abstractmethod
import numpy as np

class MetricStrategy(ABC):
    """
    An abstract base class that defines a common interface for metric calculation strategies.

    This class is intended to be subclassed by specific metric strategies, each implementing
    its own logic for calculating a metric from a confusion matrix. The class encapsulates
    common functionalities needed across different metric types, such as handling class labels 
    and printing metrics.

    Attributes:
        _classes_dict (dict): A dictionary mapping class indices to class names, inverted from
            the input labels dictionary for easier access during metric calculation.
        _metric_type (str): A string representing the type of metric this strategy calculates,
            to be set by subclasses.
    
    Methods:
        calculate: Abstract method to be implemented by subclasses for metric calculation.
        print_metric: Prints the calculated metric in a human-readable format.
    """
    def __init__(self, labels_dict: dict = None):
        """
        Initializes the MetricStrategy with a labels dictionary, if provided.

        Params:
            labels_dict (dict, optional): A dictionary mapping class names to label indices.
                If provided, it's inverted to map indices to names for easier metric calculation.
        """
        # Inverts the labels_dict to create a mapping from class indices to class names. This inversion
        # is crucial for metric calculation and result interpretation, as it allows metrics to be reported
        # and understood in terms of class names rather than numerical indices, enhancing readability and
        # clarity when analyzing model performance.
        self._classes_dict = {value: key for key, value in labels_dict.items()}
        
        self._metric_type = None

    @abstractmethod
    def calculate(self,
                  confmatrix: np.ndarray,
                  as_percentage: bool = True,
                  as_dict: bool = True):
        """
        Abstract method to calculate the metric from a confusion matrix. This method must be
        implemented by subclasses to define specific metric calculations.

        Params:
            confmatrix (np.ndarray): The confusion matrix from which to calculate the metric.
            as_percentage (bool, optional): Whether to return the metric value as a percentage.
                Defaults to True.
            as_dict (bool, optional): Whether to return the metric in a dictionary format,
                with the metric type and value(s). Defaults to True.
        
        Returns:
            The calculated metric, with the format depending on `as_percentage` and `as_dict`.
        """
        pass

    def print_metric(self,
              confmatrix: np.ndarray,
              as_percentage: bool = True,
              decimal_points: int = 4):
        """
        Prints the calculated metric in a human-readable format, including the metric type
        and value, formatted according to the specified number of decimal points.

        Params:
            confmatrix (np.ndarray): The confusion matrix from which to calculate the metric.
            as_percentage (bool, optional): Whether the metric values are displayed as percentages.
                Defaults to True.
            decimal_points (int, optional): The number of decimal places for the metric values.
                Defaults to 4.
        """
        # Calculate the metric using the provided confusion matrix, with results as a dictionary
        metric_dict = self.calculate(confmatrix=confmatrix,
                                  as_percentage=as_percentage,
                                  as_dict=True)
        
        # Print the metric type in a formatted header
        print("__________", metric_dict["metric_type"].capitalize(), "__________")

        # Remove the metric type from the dictionary to print only the values
        del metric_dict["metric_type"]

        # Iterate through the metric dictionary, printing each key-value pair
        for key, value in metric_dict.items():
            print(f"{key}: {value:.{decimal_points}f}%")
