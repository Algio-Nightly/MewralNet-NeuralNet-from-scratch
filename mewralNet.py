import numpy as np
import csv
import pickle

activation_file_path = "model/activations.csv"

class MewralNet():
    def __init__(self, layer_dims, weights_file=None):
        if weights_file == None:
            self.layer_dims = layer_dims
            self.weights, self.biases = self.initialize_weights_biases(layer_dims)
            self.activations = []

    def layer(self,input_layer_activations, layer_weights, biases) -> list:
        new_layer = np.dot(layer_weights, input_layer_activations) + biases
        return new_layer


    def write_to_file(self,input:list):
        with open(activation_file_path, 'a') as act_file:
            writer = csv.writer(act_file)
            writer.writerow(input)
        pass

    def sigmoid(self,array):
        # Sigmoid = 1 / 1 + e^(-x)
        return 1/(1+np.exp(-array))

    def initialize_weights_biases(self, layer_dims):
        weights = []
        biases = []
        for dim in range(0, len(layer_dims)-1):
            input_size = layer_dims[dim]
            output_size = layer_dims[dim+1]
            weights.append(np.random.uniform(-1, 1, (output_size, input_size)))
            biases.append(np.zeros(output_size))
        return (np.array(weights, dtype=object), np.array(biases, dtype=object))
    
    def save_weights(self):
        name = "_1"
        file_path = f"model/weights{name}.weights"
        biases_file_path = f"model/biases{name}.biases"

        with open(file=file_path, mode="wb") as weight_storage:
            pickle.dump(self.weights, weight_storage)
        
        with open(biases_file_path, "wb") as biases_storage:
            pickle.dump(self.biases, biases_storage)

    def predict(self, input):
        current_layer = np.array(input)
        self.activations = [current_layer]
        self.write_to_file(current_layer)
        for dim in range(0, len(self.layer_dims)-1):
            current_layer = np.round(self.sigmoid(self.layer(current_layer, layer_weights=self.weights[dim], biases=self.biases[dim])),4)
            self.activations.append(current_layer)
            self.write_to_file(current_layer)
        self.save_weights()
        
        return self.activations[-1]


# input = np.array([0.0, 0.02, 0.041, 0.061, 0.082, 0.102, 0.122, 0.143, 0.163, 0.184,
#     0.204, 0.224, 0.245, 0.265, 0.266, 0.306, 0.327, 0.347, 0.367, 0.388,
#     0.408, 0.429, 0.449, 0.469, 0.49, 0.51, 0.531, 0.551, 0.571, 0.592,
#     0.612, 0.633, 0.653, 0.673, 0.694, 0.714, 0.735, 0.755, 0.776, 0.796,
#     0.816, 0.837, 0.857, 0.878, 0.898, 0.9212, 0.939, 0.959, 0.98, 1.0])

# layer_dims = [input.shape[0], 15, 15, 16, 2]


# newNet = MewralNet(layer_dims=layer_dims)
# newNet.predict(input)
    



