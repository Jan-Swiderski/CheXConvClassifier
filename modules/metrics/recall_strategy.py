from typing import Optional
from numpy import ndarray
from .metric_strategy import MetricStrategy

class RecallStrategy(MetricStrategy):
    labels_assignable = True
    def calculate(self,
                  confmatrix: ndarray,
                  as_percentage: bool,
                  labels_dict: Optional[dict] = None):
        recall_list = []

        for index, _ in enumerate(confmatrix):

            # Compute recall only if the denominator is not zero
            if confmatrix[:, index].sum() > 0:
                recall = confmatrix[index, index] / confmatrix[:, index].sum()
                if as_percentage:
                    recall *= 100
            else:
                recall = 0

            recall_list.append(recall)

        return recall_list