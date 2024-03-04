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
        _get_recalls: Computes the recall for each class.
        _get_precisions: Computes the precision for each class.
        get_accuracy: Computes the overall accuracy.
        get_rec_prec_lists: Returns lists of recall and precision values for all classes.
        get_rec_prec_dicts: Returns dictionaries of recall and precision values for all classes, keyed by class name.
        print_stats: Prints accuracy, recall, and precision statistics.
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
        # Ensure tensors are on CPU before converting to numpy
        if gt_labels.device.type != 'cpu':
            gt_labels.cpu()
        
        if quantized_preds.device.type != 'cpu':
            quantized_preds.cpu()
        
        # Convert tensors to numpy arrays
        gt_labels = gt_labels.detach().numpy()
        quantized_preds = quantized_preds.detach().numpy()

        # Update confusion matrix counts
        np.add.at(self.__confmatrix, (gt_labels, quantized_preds), 1)

    def _get_recalls(self,
                     as_percentage: bool = False):
        """
        Computes the recall for each class.
        
        Args:
            as_percentage (bool): Whether to return recall values as percentages.
            
        Returns:
            list: Recall values for each class.
        """

        recall_list = []

        for index, _ in enumerate(self.__confmatrix):

            # Compute recall only if the denominator is not zero
            if self.__confmatrix[:, index].sum() > 0:
                recall = self.__confmatrix[index, index] / self.__confmatrix[:, index].sum()
                if as_percentage:
                    recall *= 100
            else:
                recall = 0

            recall_list.append(recall)

        return recall_list
    
    def _get_precisions(self,
                        as_percentage:bool = False):
        """
        Computes the precision for each class.
        
        Args:
            as_percentage (bool): Whether to return precision values as percentages.
            
        Returns:
            list: Precision values for each class.
        """
        precision_list = []

        for index, _ in enumerate(self.__confmatrix):
            # Compute precision only if the denominator is not zero
            if self.__confmatrix[index, :].sum() > 0:
                precision = self.__confmatrix[index, index] / self.__confmatrix[index, :].sum()
                if as_percentage:
                    precision *= 100
            else:
                precision = 0
 
            precision_list.append(precision)

        return precision_list
    
    def get_accuracy(self,
                     as_percentage:bool = False):
        """
        Computes the overall accuracy.
        
        Args:
            as_percentage (bool): Whether to return the accuracy as a percentage.
            
        Returns:
            float: The accuracy of the predictions.
        """
        accuracy = self.__confmatrix.trace() / self.__confmatrix.sum()
        if as_percentage:
            accuracy *= 100
        return accuracy
    
    def get_rec_prec_lists(self,
                           as_percentage:bool = False):
        """
        Returns lists of recall and precision values for all classes.
        
        Args:
            as_percentage (bool): Whether to return values as percentages.
            
        Returns:
            tuple of list: Lists of recall and precision values.
        """
        recall_list = self._get_recalls(as_percentage)
        precision_list = self._get_precisions(as_percentage)

        return recall_list, precision_list
    
    def get_rec_prec_dicts(self,
                           as_percentage:bool = False):
        """
        Returns dictionaries of recall and precision values for all classes, keyed by class name.
        
        Args:
            as_percentage (bool): Whether to return values as percentages.
            
        Returns:
            tuple of dict: Dictionaries of recall and precision values.
        """
        recall_dict = {}
        precision_dict = {}

        recall_list, precision_list = self.get_rec_prec_lists(as_percentage)
        
        # Map recall and precision values to class names
        for index, value in enumerate(recall_list):
            recall_dict[self.__classes_dict[index]] = value

        for index, value in enumerate(precision_list):
            precision_dict[self.__classes_dict[index]] = value

        return recall_dict, precision_dict
    
    def print_stats(self,
                    as_percentage:bool = False):
        """
        Prints accuracy, recall, and precision statistics.
        
        Args:
            as_percentage (bool): Whether to print values as percentages.
        """
        accuracy = self.get_accuracy(as_percentage)
        recall_dict, precision_dict = self.get_rec_prec_dicts(as_percentage)

        # Print formatted statistics
        if as_percentage:
            print(f"Accuracy: {accuracy:.4f}%")
            for i in range(self.__num_classes):
                class_name = self.__classes_dict[i]
                print(f"Recall for class {class_name}: {recall_dict[class_name]:.4f}%")
                print(f"Precision for class {class_name}: {precision_dict[class_name]:.4f}%")
        else:
            print(f"Accuracy: {accuracy}")
            for i in range(self.__num_classes):
                class_name = self.__classes_dict[i]
                print(f"Recall for class {class_name}: {recall_dict[class_name]:.6f}")
                print(f"Precision for class {class_name}: {precision_dict[class_name]:.6f}")
