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
            print(f"Output: {output} , Expected: {expected_value}")


    # Will return the final output of the multilayered perceptron
    def feed_forward_pass(self, initial_input):
        layer_output = initial_input

        for layer in self.layers:
            layer_output = layer.feed_forward(layer_output)

        return layer_output[0]