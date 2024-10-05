from abc import ABC, abstractmethod

class Metrics(ABC):

    @classmethod
    @abstractmethod
    def get_metric(cls,true_positive,false_positive,true_negative,false_negative):
        pass