from abc import ABC, abstractmethod
from metrics import ABC

class Recall(ABC):

    @classmethod
    def get_metrics(cls,true_positive,true_negative,false_positive,false_negative):
        return (true_positive/(true_positive + false_negative))