from abc import ABC, abstractmethod

class Optimizer(ABC):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self, layer, gradients, learning_rate):
        pass