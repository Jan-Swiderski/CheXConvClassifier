"""
This module defines the AccuracyStrategy class, which implements the metric calculation strategy 
for computing accuracy from a confusion matrix. The class extends the MetricStrategy abstract 
base class, providing a concrete implementation of the calculation method specific to accuracy.
Accuracy is defined as the ratio of correctly predicted observations to the total observations 
and is a common metric for evaluating classification models.
"""
import numpy as np
from .metric_strategy import MetricStrategy

class AccurracyStrategy(MetricStrategy):
    """
    A concrete class for calculating accuracy, extending the MetricStrategy abstract base class.

    This class implements the calculation of accuracy from a confusion matrix, providing 
    both the raw value and percentage form of the metric. It utilizes the confusion matrix's
    trace and sum to compute the ratio of correct predictions to total predictions.

    Attributes:
        _metric_type (str): Specifies the type of metric ('accuracy') calculated by this strategy.
    """
    def __init__(self, labels_dict: dict = None):
        """
        Initializes the AccuracyStrategy with an optional labels dictionary.

        The labels dictionary is passed to the base class for creating a mapping from class indices
        to names, enhancing readability of the results.

        Params:
            labels_dict (dict, optional): A dictionary mapping class names to label indices.
        """
        super().__init__(labels_dict)
        self._metric_type = "accuracy"

    def calculate(self,
                  confmatrix: np.ndarray,
                  as_percentage: bool,
                  as_dict: bool):
        """
        Calculates the accuracy based on the given confusion matrix.

        Accuracy is computed as the ratio of the trace of the confusion matrix (sum of correct predictions)
        to the total number of predictions (sum of all elements in the confusion matrix).

        Params:
            confmatrix (np.ndarray): The confusion matrix from which to calculate accuracy.
            as_percentage (bool): Whether to return the accuracy value as a percentage.
            as_dict (bool): Whether to return the accuracy in a dictionary format.

        Returns:
            If as_dict is True, returns a dictionary with the metric type and accuracy.
            Otherwise, returns the raw accuracy value or percentage, based on as_percentage.
        """
        # Calculate raw accuracy as the ratio of correct predictions (matrix trace) to total predictions
        accuracy = confmatrix.trace() / confmatrix.sum()
        # Convert accuracy to percentage if requested
        if as_percentage:
            accuracy *= 100

        # Return accuracy in the requested format (dictionary or raw/percentage)
        if as_dict:
            return {"metric_type": self._metric_type,
                    "Accuracy": accuracy}
        else:
            return accuracy