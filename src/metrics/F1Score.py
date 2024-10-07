from abc import ABC, abstractmethod
from src.metrics.Metrics import Metrics
from src.metrics.Precision import Precision
from src.metrics.Recall import Recall

class F1Score(Metrics):

    @classmethod
    def get_metric(cls,true_positive,true_negative,false_positive,false_negative):
        precision_value = Precision.get_metric(true_positive,true_negative,false_positive,false_negative)
        recall_value = Recall.get_metric(true_positive,true_negative,false_positive,false_negative)

        return ( (2 * precision_value * recall_value) / (precision_value + recall_value ) )