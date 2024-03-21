"""
This module defines the PrecisionStrategy class, a concrete implementation of the MetricStrategy 
abstract base class, designed for calculating the precision metric from a confusion matrix. 
Precision, a key metric in classification tasks, measures the proportion of true positive predictions 
among all positive predictions made by the model. This class enables the calculation of precision 
for each class individually, as well as the overall precision, and supports returning results 
either as raw values or percentages, and optionally in a dictionary format for enhanced readability.
"""
import numpy as np
from .metric_strategy import MetricStrategy

class PrecisionStrategy(MetricStrategy):
    """
    Implements precision calculation as part of the metric strategies for evaluating classification models.

    Precision is defined as the ratio of true positives to the sum of true and false positives, indicating
    the accuracy of positive predictions. This class provides functionality to compute precision for each
    class and overall precision, based on a provided confusion matrix.

    Attributes:
        _metric_type (str): Identifies 'precision' as the type of metric calculated by this strategy.
    """
    def __init__(self, labels_dict = None):
        """
        Initializes the PrecisionStrategy. 
        Leaving the labels_dict as None will result in improper working of this class!

        Params:
            labels_dict (dict): A dictionary mapping class names to their respective label indices.
        """
        super().__init__(labels_dict)
        self._metric_type = "precision"
    def calculate(self,
                  confmatrix: np.ndarray,
                  as_percentage: bool,
                  as_dict: bool):
        """
        Calculates precision for each class based on the provided confusion matrix. Precision is computed 
        as the ratio of true positives (diagonal elements) to the sum of true and false positives (column sums).

        Params:
            confmatrix (np.ndarray): The confusion matrix from which to calculate precision.
            as_percentage (bool): Specifies whether to return the precision values as percentages.
            as_dict (bool): Determines whether to return precision values in a dictionary format,
                mapping class names to their precision values.

        Returns:
            A list of precision values for each class or a dictionary of class names to precision values,
            depending on the value of as_dict. If as_percentage is True, precision values are multiplied by 100.
        """
        precision_list = []

        # Iterate through each class (column) in the confusion matrix
        for index, _ in enumerate(confmatrix):
            # Check if there are any positive predictions for this class to avoid division by zero
            if confmatrix[:, index].sum() > 0:
                # Calculate precision as the ratio of true positives to all positive predictions (column sum)
                precision = confmatrix[index, index] / confmatrix[:, index].sum()
                # Convert precision to percentage if requested
                if as_percentage:
                    precision *= 100
            else:
                # Set precision to 0 if there are no positive predictions for this class
                precision = 0

            precision_list.append(precision)

        # If requested, return precision values in a dictionary format mapping class names to their precision values
        if as_dict:
            precision_dict = {"metric_type": self._metric_type}
            for index, value in enumerate(precision_list):
                # Use the class names from _classes_dict for keys
                precision_dict[self._classes_dict[index]] = value
            return precision_dict
        
        else:
            # Otherwise, return the list of precision values directly
            return precision_list

