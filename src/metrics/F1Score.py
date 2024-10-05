from abc import ABC, abstractmethod
from metrics import ABC

class F1Score(ABC):

    @classmethod
    def get_metrics(cls,true_positive,true_negative,false_positive,false_negative):
        precision_value = Precision.get_metrics(true_positive,true_negative,false_positive,false_negative)
        recall_value = Recall.get_metrics(true_positive,true_negative,false_positive,false_negative)

        return ( (2 * precision_value * recall_value) / (precision_value + recall_value ) )