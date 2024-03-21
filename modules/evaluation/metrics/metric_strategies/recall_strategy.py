"""
This module provides the RecallStrategy class, a concrete implementation of the MetricStrategy 
abstract base class, tailored for calculating the recall metric from a confusion matrix. Recall, 
also known as sensitivity, measures the proportion of actual positives that are correctly identified 
by the model. This class facilitates the computation of recall for individual classes as well as 
the overall recall, allowing for results to be output as either raw values or percentages, and, 
optionally, in a dictionary format for ease of interpretation.
"""
import numpy as np
from .metric_strategy import MetricStrategy

class RecallStrategy(MetricStrategy):
    """
    Implements recall calculation as a metric strategy for evaluating classification models.

    Recall (sensitivity) is defined as the ratio of true positives to the sum of true positives 
    and false negatives, reflecting the model's ability to correctly identify all relevant instances. 
    This class provides the functionality to calculate recall for each class based on a provided 
    confusion matrix.

    Attributes:
        _metric_type (str): Specifies 'recall' as the type of metric calculated by this strategy.
    """
    def __init__(self, labels_dict: dict = None):
        """
        Initializes the RecallStrategy .
        Leaving the labels_dict as None will result in improper working of this class!

        Params:
            labels_dict (dict): A dictionary mapping class names to their respective label indices.
        """
        super().__init__(labels_dict)
        self._metric_type = "recall"
    def calculate(self,
                  confmatrix: np.ndarray, 
                  as_percentage: bool,
                  as_dict: bool):
        """
        Calculates recall for each class using the provided confusion matrix. Recall is computed as
        the ratio of true positives (diagonal elements) to the sum of true positives and false negatives
        (row sum).

        Params:
            confmatrix (np.ndarray): The confusion matrix from which to calculate recall.
            as_percentage (bool): Specifies whether to return the recall values as percentages.
            as_dict (bool): Determines whether to return recall values in a dictionary format,
                mapping class names to their recall values.

        Returns:
            A list of recall values for each class or a dictionary mapping class names to recall values,
            depending on the value of as_dict. If as_percentage is True, recall values are multiplied by 100.
        """
        recall_list = []
        # Iterate through each class (row) in the confusion matrix
        for index, _ in enumerate(confmatrix):
            # Calculate recall if there are actual positives (avoid division by zero)
            if confmatrix[index, :].sum() > 0:
                recall = confmatrix[index, index] / confmatrix[index, :].sum()
                # Convert recall to percentage if requested
                if as_percentage:
                    recall *= 100
            else:
                # Set recall to 0 if there are no actual positives for this class
                recall = 0
 
            recall_list.append(recall)

        # Return recall values in the requested format (list or dictionary)
        if as_dict:
            recall_dict = {"metric_type": self._metric_type}
            for index, value in enumerate(recall_list):
                # Use the class names from _classes_dict for keys
                recall_dict[self._classes_dict[index]] = value
            return recall_dict
        else:
            return recall_list
