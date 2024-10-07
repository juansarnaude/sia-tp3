from abc import ABC, abstractmethod
from src.metrics.Metrics import Metrics

class Precision(Metrics):

    @classmethod
    def get_metric(cls,true_positive,true_negative,false_positive,false_negative):
        return ( true_positive/(true_positive + false_positive) )