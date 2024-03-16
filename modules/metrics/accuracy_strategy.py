from numpy import ndarray
from .metric_strategy import MetricStrategy

class AccurracyStrategy(MetricStrategy):
    def calculate(self,
                  confmatrix: ndarray,
                  as_percentage: bool = False):
        accuracy = confmatrix.trace() / confmatrix.sum()
        if as_percentage:
            accuracy *= 100
        return accuracy
    def print_metric(self,
              confmatrix: ndarray,
              as_percentage: bool = False):
        accuracy = self.calculate(confmatrix=confmatrix,
                                  as_percentage=as_percentage)
        
        if as_percentage:
            print(f"Accuracy: {accuracy}%")
        else:    
            print(f"Accuracy: {accuracy}")
        