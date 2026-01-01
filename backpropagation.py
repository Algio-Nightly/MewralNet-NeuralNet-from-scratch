import numpy as np
import math

from mewralNet import MewralNet

class Backpropagation(MewralNet):
    def __init__(self, layer_dims, learning_rate, weights_file = None):
        super().__init__(layer_dims, weights_file=None)
        self.learning_rate = learning_rate
        self.desired_activations = []
        self.loss_per_epoch = []

    def mean_sqared_error(self, actual_value, desired_value):
        return math.pow((desired_value - actual_value), 2)

    def cost_function(self, current_activations, desired_activations) -> np.ndarray:
        return np.mean((desired_activations-current_activations)**2)
    
    def sigmoid_derivative(self, array):
        s = super().sigmoid(array)
        return s * (1-s)
    
    def backpropagate(self, train_dataset, desired_outputs, epocs = 1000):
        classes  = np.unique(desired_outputs)
        desired_outputs_matrix = np.zeros((len(train_dataset),len(classes)))
        desired_outputs_matrix[np.arange(len(train_dataset)),desired_outputs] = 1

        for i in range(epocs):
            total_epocs_loss = 0
            for train_example, desired_output in zip(train_dataset, desired_outputs_matrix):
                self.predict(train_example)
                self.desired_activations = desired_output
                cost_array = self.cost_function(self.activations[-1], self.desired_activations)
                total_epocs_loss += cost_array.mean()
                self.nudge_weights()
            self.loss_per_epoch.append(total_epocs_loss/len(train_dataset))
        pass

    
         


    def nudge_weights(self, i=1):   
        a_current = self.activations[i]
        a_prev = self.activations[i-1]
        z = np.dot(self.weights[i-1], a_prev) + self.biases[i-1]
        
        if i==len(self.layer_dims)-1:
            print("Attained Base Case, Backpropagating....")
            delta = self.sigmoid_derivative(z)*2*(a_current-self.desired_activations)
            delta = self.sigmoid_derivative(z)*2*(a_current-self.desired_activations)

            weight_derivative = np.outer(delta,a_prev) #dC/dw
            bias_derivative = 1*delta #dz/db is just 1 :) so dC/db is just delta 
            activation_derivative = np.dot(self.weights[i-1].T, delta) # dC/da
            # modifying weights
            self.weights[i-1] -= weight_derivative*self.learning_rate
            self.biases[i-1] -= bias_derivative*self.learning_rate
            # returning dC/da
            return activation_derivative
        
        else:
            print(f"Trying to nudge weights in layer{i}")
            forward_error = self.nudge_weights(i+1) #catching dC/da 
            print(f"Backpropagation completeted in layer {i}")
            delta = self.sigmoid_derivative(z)*forward_error 

            weight_derivative = np.outer(delta, a_prev)
            bias_derivative = 1*delta #dz/db is just 1 :) so dC/db is just delta 
            activation_derivative = np.dot(self.weights[i-1].T, delta)

            self.weights[i-1]-= weight_derivative*self.learning_rate
            self.biases[i-1] -= bias_derivative*self.learning_rate
            return activation_derivative
        
