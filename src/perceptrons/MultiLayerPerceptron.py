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
            else:
                self.layers.append(Layer(layers[i], layers[i-1], learning_rate))


    def run(self, dataset, periods, epsilon):
        for i in range(periods):
            data_list = dataset.values.tolist()
            delta_w_cum_list = None

            for data in data_list:
                expected_value = data.pop()

                output = self.feed_forward_pass(data)
                error = self.error(output[-1][0], expected_value )
                print(f"Output: {output[-1][0]} , Expected: {expected_value}, Error: {error}")

                delta_w = self.backwards_propagation(output, expected_value)

                if delta_w_cum_list is None:
                    delta_w_cum_list = [np.zeros_like(dw) for dw in delta_w]
                else:
                    delta_w_cum_list = [arr1 + arr2 for arr1, arr2 in zip(delta_w_cum_list, delta_w)]

            print("--------------")
            self.update_weights(delta_w_cum_list)


    # Will return the final output of the multilayered perceptron
    def feed_forward_pass(self, initial_input):
        layer_output = initial_input
        output = []
        for layer in self.layers:
            layer_output = layer.feed_forward(layer_output)
            output.append(layer_output)     # The initial input is saved, maybe we shouldn't do that
        return output



    # output: list that contains a list of outputs in each layer.
    # error: error between expected and result
    def backwards_propagation(self, outputs, expected_output):
        delta = (np.array(expected_output) - outputs[-1]) * self.layers[-1].get_theta_diff()
        delta_w_list = [np.outer(delta, outputs[-2]) * self.learning_rate]

        index = len(self.layers)-2
        while index > 0:
            list_w_next_layer = self.layers[index+1].get_w_list() # Is this list format correct
            delta = np.matmul(delta, list_w_next_layer)  * self.layers[index].get_theta_diff()
            delta_w_list.append( np.outer(delta, outputs[index-1]) * self.learning_rate)
            index -= 1

        return delta_w_list
        #print(delta_w_list)


    def update_weights(self, delta_w_list):
        index = len(self.layers) - 1
        for delta_w_layer in delta_w_list:
            self.layers[index].update_w(delta_w_layer)
            index -= 1

    def error(self, expected_value, computed_value):
        return (expected_value-computed_value)**2