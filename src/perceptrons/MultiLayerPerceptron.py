import numpy as np

from src.models.InputLayer import InputLayer
from src.models.Layer import Layer


class MultiLayerPerceptron:
    def __init__(self, layers, learning_rate, input_size):
        self.layers = []
        self.learning_rate = learning_rate

        # We are going to add to the layers the input layer so that its easier to implement
        layers.insert(0, input_size)

        for i in range(len(layers)):
            if i == 0:
                # This layer will just return the input array as it is.
                self.layers.append(InputLayer())

            self.layers.append(Layer(layers[i], layers[i-1], learning_rate))


    def run(self, dataset, periods, epsilon):
        data_list = dataset.values.tolist()

        for data in data_list:
            expected_value = data.pop()

            output = self.feed_forward_pass(data)
            error = self.error(output[-1][0], expected_value )
            print(f"Output: {output[-1][0]} , Expected: {expected_value}, Error: {error}")


            self.backwards_propagation(output, error, data)



    # Will return the final output of the multilayered perceptron
    def feed_forward_pass(self, initial_input):
        layer_output = initial_input
        for layer in self.layers:
            layer_output = layer.feed_forward(layer_output)
        return layer_output[0]


    def backwards_propagation(self, outputs_layer_list, error, initial_input):
        theta_diff = self.layers[len(self.layers)-1].get_theta_diff()

        delta = np.matmul(error, np.diag(theta_diff)) * self.learning_rate

        # n-2 layer values
        values = self.layers[len(self.layers)-2].last_computed_output

        delta_w_list = np.matmul(np.split(delta,len(delta)), np.split(values,1))
        prev_w_list = self.layers[len(self.layers)-1].get_w_list()

        i = len(self.layers)-2
        while i >= 0:
            if i == 0:
                values = np.insert(initial_input)
            else:
                values = self.layers[i].last_computed_output

            delta_w = self.layers[i].last_computed_output

            i -= 1










    def error(self, expected_value, computed_value):
        return (expected_value - computed_value)**2