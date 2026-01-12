import numpy as np
from mewralNet import MewralNet
class RecurrentMewralNet:
    def __init__(self, input_size, hidden_size, output_size, initialzation = "random"):
        RNNdims = []
        self.hidden_size = hidden_size
        if initialzation == "random":
            self.Weights_xh = np.random.randn(input_size, hidden_size) / 1000
            self.Weights_hh = np.random.randn(hidden_size, hidden_size) / 1000
            self.Weights_hy = np.random.randn(hidden_size, output_size) / 1000
        else:
            self.Weights_xh = np.random.randn(input_size, hidden_size) * np.sqrt(1.0/input_size)
            self.Weights_hh = np.random.randn(hidden_size, hidden_size) * np.sqrt(1.0/hidden_size)
            self.Weights_hy = np.random.randn(hidden_size, output_size) * np.sqrt(1.0/hidden_size)

        self.Biases_hh = np.zeros((1,hidden_size))
        self.Biases_hy = np.zeros((1,output_size))

    def BPTT(self, inputs, targets, learning_rate = 0.05):
        outputs, hidden = self.forward(inputs)
        
        loss = 0
        for t in range(len(outputs)):
            loss += 0.5 * (outputs[t] - targets[t]) ** 2
        # print(loss)

        GA_Weights_xh,GA_Weights_hh,GA_Weights_hy, GA_biases_h, GA_biases_y= self.backward(inputs, outputs, targets, hidden)

        for grad in [GA_Weights_xh, GA_Weights_hh, GA_Weights_hy]:
            np.clip(grad, -5, 5, out=grad)


        self.Weights_xh -= learning_rate * GA_Weights_xh
        self.Weights_hh -= learning_rate * GA_Weights_hh
        self.Weights_hy -= learning_rate * GA_Weights_hy

        self.Biases_hh -= learning_rate * GA_biases_h
        self.Biases_hy -= learning_rate * GA_biases_y

        return np.mean(loss)
 
    def forward(self, inputs, hidden_state = None):
        hidden = {}

        if hidden_state is None:
            hidden[-1] = np.zeros((1, self.hidden_size))
        else : hidden[-1] = hidden_state

        output = []

        for t in range(len(inputs)):
            x_t = np.array([inputs[t]])
            prev_hidden_layer = hidden[t-1]

            current_h = np.tanh(np.dot(x_t, self.Weights_xh)+np.dot(prev_hidden_layer, self.Weights_hh)+self.Biases_hh)
            hidden[t] = current_h

            y_t = np.dot(current_h, self.Weights_hy) +self.Biases_hy
            output.append(y_t)
        
        return output, hidden
    
    def backward(self, inputs:np.ndarray, outputs, targets, hidden):
        GA_Weights_xh = np.zeros_like(self.Weights_xh)
        GA_Weights_hh = np.zeros_like(self.Weights_hh)
        GA_Weights_hy = np.zeros_like(self.Weights_hy)

        GA_bias_h = np.zeros_like(self.Biases_hh)
        GA_bias_y = np.zeros_like(self.Biases_hy)
        
        h_next = np.zeros((1, self.hidden_size))
        
        for t in reversed(range(len(inputs))):
            loss_derivative = outputs[t] - targets[t] #dL/dW_yt

            # Gradient for Hidden -> Output Weights
            GA_Weights_hy += np.dot(hidden[t].T, loss_derivative)

            # Gradient for Hidden -> HIdden Weights
            h_prev = hidden[t-1]
            raw = np.dot(loss_derivative, self.Weights_hy.T)
            h_total = raw + h_next
            h_activation = h_total * (1 - (hidden[t] ** 2))
            GA_Weights_hh += np.dot(h_prev.T, h_activation)
            
            # Gradient for Input -> Hidden Weights
            x_t = inputs[t].reshape(1, -1)
            GA_Weights_xh += np.dot(x_t.T, h_activation)

            h_next = np.dot(h_activation, self.Weights_hh.T)

            #Bias updates
            GA_bias_y += loss_derivative
            GA_bias_h += h_activation

        return GA_Weights_xh, GA_Weights_hh, GA_Weights_hy, GA_bias_h, GA_bias_y
    
    def train():
        pass

