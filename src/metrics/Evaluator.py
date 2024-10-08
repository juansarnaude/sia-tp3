import numpy as np

from src.perceptrons.MultiLayerPerceptron import MultiLayerPerceptron
from src.metrics.Accuracy import Accuracy
from src.metrics.F1Score import F1Score
from src.metrics.Precision import Precision
from src.metrics.Recall import Recall

class Evaluator:
    def __init__(self, input_set_len, output_set_len, epsilon, normalize_prediction):
        self.input_set_len = input_set_len
        self.output_set_len = output_set_len
        self.epsilon = epsilon
        self.normalize_prediction = normalize_prediction

    def evaluate(self, perceptron, training_set, training_expected_values, testing_set, testing_expected_values, epochs: int):
        
        ### Training
        perceptron.train(training_set, training_expected_values, epochs, self.epsilon)

        ### Testing
        confusion_matrix = np.zeros((self.input_set_len, self.output_set_len))
        
        for input, expected_value in zip(testing_set, testing_expected_values):
            prediction = self.normalize_prediction(perceptron.predict(input))
            confusion_matrix[expected_value][prediction] += 1

        
        # compute the confusion matrix results
        true_positive=0
        true_negative=0
        false_positive=0
        false_negative=0

        for i in range(len(confusion_matrix)):
            for j in range(len(confusion_matrix[i])):
                if i == j:
                    true_positive += confusion_matrix[i][j]
                else:
                    false_negative += confusion_matrix[i][j]
                    true_negative += confusion_matrix[j][j]
                    false_positive += confusion_matrix[j][i]

        accuracy = Accuracy.get_metric(true_positive,true_negative,false_positive,false_negative)
        f1 = F1Score.get_metric(true_positive,true_negative,false_positive,false_negative)
        precision = Precision.get_metric(true_positive,true_negative,false_positive,false_negative)
        recall = Recall.get_metric(true_positive,true_negative,false_positive,false_negative)

        return accuracy, f1, precision, recall
        



