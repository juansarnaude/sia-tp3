from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Callable, Optional, Union, Tuple
import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingMetrics:
    """Stores metrics from training process"""
    epoch: int
    training_error: float
    testing_error: Optional[float] = None

class Neuron:
    """Represents a single neuron with weights"""
    def __init__(self, input_size: int, weight_range: Tuple[float, float] = (-0.1, 0.1)):
        self.weights = np.random.uniform(*weight_range, size=input_size)
    
    def get_weighted_sum(self, inputs: np.ndarray, bias: float) -> float:
        return np.dot(self.weights, inputs) + bias
    
    def update_weights(self, delta: np.ndarray) -> None:
        self.weights += delta

class Perceptron(ABC):
    """Abstract base class for perceptron implementations"""
    
    def __init__(self, 
                 input_size: int,
                 learning_rate: float = 0.01,
                 epsilon: float = 1e-4,
                 weight_range: Tuple[float, float] = (-0.1, 0.1)):
        """
        Initialize perceptron.
        
        Args:
            input_size: Number of input features
            learning_rate: Learning rate for weight updates
            epsilon: Convergence threshold
            weight_range: Range for initial weight randomization
        """
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.neuron = Neuron(input_size, weight_range)
        self.bias = np.random.uniform(*weight_range)
        
        # Data normalization parameters
        self.data_min: Optional[float] = None
        self.data_max: Optional[float] = None
        
        # Training history
        self.history: List[TrainingMetrics] = []

    def predict(self, x: np.ndarray) -> float:
        """Make prediction for input x"""
        if x.shape[0] != self.input_size:
            raise ValueError(f"Expected input size {self.input_size}, got {x.shape[0]}")
        
        weighted_sum = self.neuron.get_weighted_sum(x, self.bias)
        theta = self.theta(weighted_sum)
        return self.compute_activation(theta)

    def fit(self, 
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_test: Optional[np.ndarray] = None,
            y_test: Optional[np.ndarray] = None,
            epochs: int = 1000,
            shuffle: bool = True) -> List[TrainingMetrics]:
        """
        Train the perceptron.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Optional test features
            y_test: Optional test labels
            epochs: Maximum number of training epochs
            shuffle: Whether to shuffle data each epoch
            
        Returns:
            List of training metrics per epoch
        """
        # Initialize data normalization
        self._init_normalization(y_train, y_test)
        
        # Convert inputs to numpy arrays if needed
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        n_samples = len(X_train)
        indices = np.arange(n_samples)
        
        self.history = []
        
        for epoch in range(epochs):
            if shuffle:
                np.random.shuffle(indices)
                
            # Training phase
            epoch_predictions = []
            for idx in indices:
                x, y = X_train[idx], y_train[idx]
                pred = self._train_step(x, y)
                epoch_predictions.append(pred)
            
            # Calculate errors
            train_error = self.error(epoch_predictions, y_train)
            
            # Test if validation data provided
            test_error = None
            if X_test is not None and y_test is not None:
                test_predictions = [self.predict(x) for x in X_test]
                test_error = self.error(test_predictions, y_test)
            
            # Store metrics
            metrics = TrainingMetrics(epoch=epoch, 
                                    training_error=train_error,
                                    testing_error=test_error)
            self.history.append(metrics)
            
            # Log progress
            log_msg = f"Epoch {epoch}: train_error={train_error:.6f}"
            if test_error is not None:
                log_msg += f", test_error={test_error:.6f}"
            # logger.info(log_msg)
            
            # Check convergence
            if train_error <= self.epsilon:
                logger.info(f"Converged at epoch {epoch}")
                break
                
        return self.history

    def _train_step(self, x: np.ndarray, y: float) -> float:
        """Perform single training step"""
        weighted_sum = self.neuron.get_weighted_sum(x, self.bias)
        prediction = self.predict(x)
        
        # Update weights and bias
        weight_update = self.delta_w(prediction, y, x, weighted_sum)
        self.neuron.update_weights(weight_update)
        
        bias_update = self.delta_b(prediction, y, weighted_sum)
        self.bias += bias_update
        
        return prediction

    def _init_normalization(self, y_train: np.ndarray, y_test: Optional[np.ndarray] = None) -> None:
        """Initialize normalization parameters"""
        all_y = y_train if y_test is None else np.concatenate([y_train, y_test])
        self.data_min = np.min(all_y)
        self.data_max = np.max(all_y)

    def save(self, path: Union[str, Path]) -> None:
        """Save model parameters"""
        path = Path(path)
        state = {
            'weights': self.neuron.weights.tolist(),
            'bias': float(self.bias),
            'data_min': self.data_min,
            'data_max': self.data_max,
            'input_size': self.input_size,
            'learning_rate': self.learning_rate,
            'epsilon': self.epsilon
        }
        
        with open(path, 'w') as f:
            json.dump(state, f)
            
    @classmethod
    def load(cls, path: Union[str, Path]):
        """Load model parameters"""
        path = Path(path)
        with open(path, 'r') as f:
            state = json.load(f)
            
        instance = cls(
            input_size=state['input_size'],
            learning_rate=state['learning_rate'],
            epsilon=state['epsilon']
        )
        
        instance.neuron.weights = np.array(state['weights'])
        instance.bias = state['bias']
        instance.data_min = state['data_min']
        instance.data_max = state['data_max']
        
        return instance

    @abstractmethod
    def theta(self, weighted_sum: float) -> float:
        """Apply theta function"""
        pass
    
    @abstractmethod
    def compute_activation(self, theta_value: float) -> float:
        """Apply activation function"""
        pass
    
    @abstractmethod
    def theta_diff(self, h: float) -> float:
        """Compute derivative of theta function"""
        pass
    
    @abstractmethod
    def delta_w(self, prediction: float, target: float, 
                x: np.ndarray, weighted_sum: float) -> np.ndarray:
        """Compute weight updates"""
        pass
    
    @abstractmethod
    def delta_b(self, prediction: float, target: float, 
                weighted_sum: float) -> float:
        """Compute bias update"""
        pass
    
    @abstractmethod
    def error(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute error metric"""
        pass
    
    def normalize(self, value: float) -> float:
        """Normalize value to output range"""
        if self.data_min is None or self.data_max is None:
            raise RuntimeError("Normalization not initialized. Run fit first.")
        return np.interp(value, 
                        self._get_activation_range(), 
                        [self.data_min, self.data_max])
    
    @abstractmethod
    def _get_activation_range(self) -> Tuple[float, float]:
        """Get range of activation function"""
        pass


class PerceptronNonLinear(Perceptron):
    """Non-linear perceptron implementation with configurable activation function"""
    
    # Available activation functions
    ACTIVATIONS = {
        'sigmoid': (
            lambda x, beta: 1 / (1 + np.exp(-beta * x)),  # function
            lambda x, beta: beta * (1 - x) * x,  # derivative
            (0, 1)  # range
        ),
        'tanh': (
            lambda x, beta: np.tanh(beta * x),
            lambda x, beta: beta * (1 - x**2),
            (-1, 1)
        )
    }
    
    def __init__(self, 
                 input_size: int,
                 learning_rate: float = 0.01,
                 epsilon: float = 1e-4,
                 beta: float = 1.0,
                 activation: str = 'sigmoid'):
        """
        Initialize non-linear perceptron.
        
        Args:
            input_size: Number of input features
            learning_rate: Learning rate for weight updates
            epsilon: Convergence threshold
            beta: Steepness parameter for activation function
            activation: Activation function type ('sigmoid' or 'tanh')
        """
        super().__init__(input_size, learning_rate, epsilon)
        
        if activation not in self.ACTIVATIONS:
            raise ValueError(f"Unknown activation: {activation}. "
                           f"Available: {list(self.ACTIVATIONS.keys())}")
        
        self.activation_name = activation
        self.beta = beta
        
        # Get activation function and its derivative
        self.activation_fn, self.activation_diff, self.activation_range = \
            self.ACTIVATIONS[activation]

    def theta(self, weighted_sum: float) -> float:
        """Identity function for weighted sum"""
        return weighted_sum
    
    def compute_activation(self, theta_value: float) -> float:
        """Apply activation function and normalize output"""
        activation = self.activation_fn(theta_value, self.beta)
        return self.normalize(activation)
    
    def theta_diff(self, h: float) -> float:
        """Compute derivative of activation function"""
        activation = self.activation_fn(h, self.beta)
        return self.activation_diff(activation, self.beta)
    
    def delta_w(self, prediction: float, target: float, 
                x: np.ndarray, weighted_sum: float) -> np.ndarray:
        """Compute weight updates using gradient descent"""
        error = target - prediction
        return self.learning_rate * error * self.theta_diff(weighted_sum) * x
    
    def delta_b(self, prediction: float, target: float, 
                weighted_sum: float) -> float:
        """Compute bias update using gradient descent"""
        error = target - prediction
        return self.learning_rate * error * self.theta_diff(weighted_sum)
    
    def error(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute mean squared error"""
        return np.mean((np.array(predictions) - np.array(targets)) ** 2)
    
    def _get_activation_range(self) -> Tuple[float, float]:
        """Get range of current activation function"""
        return self.activation_range