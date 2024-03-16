from abc import ABC, abstractmethod
import numpy as np

class MetricStrategy(ABC):
    labels_assignable = False
    @abstractmethod
    def calculate(self,
                  confmatrix: np.ndarray,
                  as_percentage: bool):
        pass