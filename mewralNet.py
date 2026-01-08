import numpy as np
import csv
import pickle



class MewralNet():
    def __init__(self, layer_dims, weights_file=None):
        if weights_file == None:
            self.layer_dims = layer_dims
            self.weights, self.biases = self.initialize_weights_biases(layer_dims)
            self.activations = []

    def layer(self,input_layer_activations, layer_weights, biases) -> list:
        new_layer = np.dot(layer_weights, input_layer_activations) + biases
        return new_layer


    # def write_to_file(self,input:list):
    #     with open(activation_file_path, 'a') as act_file:
    #         writer = csv.writer(act_file)
    #         writer.writerow(input)
    #     pass
    
    @staticmethod
    def sigmoid(array):
        clipped_array = np.clip(array, -500, 500)
        
        # Sigmoid = 1 / (1 + e^(-x))
        return 1 / (1 + np.exp(-clipped_array))
    
    def sigmoid_derivative(array):
        s = MewralNet.sigmoid(array)
        return s * (1-s)

    def initialize_weights_biases(self, layer_dims):
        weights = []
        biases = []
        for dim in range(0, len(layer_dims)-1):
            input_size = layer_dims[dim]
            output_size = layer_dims[dim+1]
            weights.append(np.random.uniform(-1, 1, (output_size, input_size)))
            biases.append(np.zeros(output_size))
        return (np.array(weights, dtype=object), np.array(biases, dtype=object))
    
    # def save_weights(self):
    #     name = "_1"
    #     file_path = f"model/weights{name}.weights"
    #     biases_file_path = f"model/biases{name}.biases"

    #     with open(file=file_path, mode="wb") as weight_storage:
    #         pickle.dump(self.weights, weight_storage)
        
    #     with open(biases_file_path, "wb") as biases_storage:
    #         pickle.dump(self.biases, biases_storage)

    def predict(self, input):
        current_layer = np.array(input)
        self.activations = [current_layer]
        # self.write_to_file(current_layer)
        for dim in range(0, len(self.layer_dims)-1):
            current_layer = np.round(self.sigmoid(self.layer(current_layer, layer_weights=self.weights[dim], biases=self.biases[dim])),4)
            self.activations.append(current_layer)
            # self.write_to_file(current_layer)
        
        return self.activations[-1]




