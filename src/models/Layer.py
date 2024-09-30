from src.perceptrons.PerceptronLinear import PerceptronLinear

class Layer:
    def __init__(self, perceptron_amount, layer_input_size, learning_rate):
        self.perceptron_list = []

        for i in range(perceptron_amount):
            # We initialize all the unused params in 0.
            self.perceptron_list.append(PerceptronLinear(layer_input_size, learning_rate))

    def feed_forward(self, layer_input):
        layer_output = []

        for perceptron in self.perceptron_list:
            layer_output.append(perceptron.value_to_feed_forward(layer_input))

        return layer_output

