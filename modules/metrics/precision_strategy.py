from typing import Optional
from numpy import ndarray
from .metric_strategy import MetricStrategy

class PrecisionStrategy(MetricStrategy):
    labels_assignable = True
    def calculate(self,
                  confmatrix: ndarray, 
                  as_percentage: bool,
                  labels_dict: Optional[dict] = None):
        precision_list = []

        for index, _ in enumerate(confmatrix):
            # Compute precision only if the denominator is not zero
            if confmatrix[index, :].sum() > 0:
                precision = confmatrix[index, index] / confmatrix[index, :].sum()
                if as_percentage:
                    precision *= 100
            else:
                precision = 0
 
            precision_list.append(precision)

        return precision_list