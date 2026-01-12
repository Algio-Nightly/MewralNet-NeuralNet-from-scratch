# import RNN
import numpy as np
from math_helpers import *
class MewralOldLSTM():
    def __init__(self, input_size, hidden_size, no_of_hidden_layers, output_size, initialization = "xavier"):
        # super().__init__(input_size, hidden_size, output_size, initialzation)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_layer_count = no_of_hidden_layers
        self.Hidden_Layers = []

        prev_layer_size = input_size
        for _ in range(no_of_hidden_layers):
            self.Hidden_Layers.append(Layers.LSTMMemoryCell(prev_layer_size, hidden_size))
            prev_layer_size = hidden_size
        
        self.Weights_hy = np.random.randn(hidden_size, output_size) * np.sqrt(1.0/input_size)
        self.Biases_hy = np.zeros(output_size)

    def BPTT(self, inputs, targets, learning_rate = 0.05):
        outputs, hidden, caches = self.forward(inputs)
        
        loss = 0
        for t in range(len(outputs)):
            loss += 0.5 * (outputs[t] - targets[t]) ** 2
        GA_weights_hy, GA_bias_y, GA_hidden_hh, GA_bias_h = self.backward(inputs, outputs, targets, hidden, caches)

        np.clip(GA_weights_hy, -5, 5, out=GA_weights_hy)
        np.clip(GA_bias_y, -5, 5, out=GA_bias_y)

        for layer_grads in GA_hidden_hh:
            for key in layer_grads:
                np.clip(layer_grads[key], -5, 5, out=layer_grads[key])

        for layer_bias_grads in GA_bias_h:
            for key in layer_bias_grads:
                np.clip(layer_bias_grads[key], -5, 5, out=layer_bias_grads[key])

        self.Weights_hy -= learning_rate * GA_weights_hy
        self.Biases_hy -= learning_rate * GA_bias_y

        layer:Layers.LSTMMemoryCell
        for index, layer in enumerate(self.Hidden_Layers):
            layer.update_cell_weights_biases(GA_hidden_hh[index], GA_bias_h[index], learning_rate)

        return np.mean(loss / len(inputs))
        

    def forward(self,inputs):
        for layer in self.Hidden_Layers:
            layer.reset_state()
        hidden = {}
        hidden[-1] = np.zeros((self.hidden_layer_count, 1, self.hidden_size))

        outputs = []

        caches = []
        for t in range(len(inputs)):
            x_t = np.array([inputs[t]])
            prev_time_hidden_layers = hidden[t-1]
            current_input = x_t

            current_hidden = []
            i=0
            layer_math_cache = []
            layer: Layers.LSTMMemoryCell
            for layer in self.Hidden_Layers:
                stack = np.concatenate((current_input, prev_time_hidden_layers[i]), axis=1)
                layer_out, cache = layer.cell_forward(stack)
                layer_math_cache.append(cache)
                current_hidden.append(layer_out)
                current_input = layer_out
                i+=1
            caches.append(layer_math_cache)
              
            hidden[t] = np.array(current_hidden)
            y_t = np.dot(current_hidden[-1], self.Weights_hy) + self.Biases_hy
            outputs.append(y_t)

        return outputs, hidden, caches
    
    def backward(self, inputs, outputs, targets, hidden, caches):
        GA_weights_hy = np.zeros_like(self.Weights_hy)
        GA_bias_y = np.zeros_like(self.Biases_hy)

        GA_hidden_hh  = [{"input_weight_gradients": np.zeros_like(x.input_weights), 
                          "input_gate_weight_gradients": np.zeros_like(x.input_gate_weights), 
                          "output_gate_weight_gradients": np.zeros_like(x.output_gate_weights)} 
                          for x in self.Hidden_Layers]
        
        GA_bias_h  = [{"input_bias_gradients": np.zeros_like(x.input_biases), 
                       "input_gate_bias_gradients": np.zeros_like(x.input_gate_biases), 
                       "output_gate_bias_gradients":np.zeros_like(x.output_gate_biases)} 
                       for x in self.Hidden_Layers]

        dh_next = np.zeros((self.hidden_layer_count, self.hidden_size))
        state_cell_next = np.zeros((self.hidden_layer_count, self.hidden_size))
        for t in reversed(range(len(inputs))):
            loss_derivative = outputs[t] - targets[t]
            # loss_derivative /= len(inputs)
            current_time_hidden = hidden[t]
            GA_weights_hy += np.dot(current_time_hidden[-1].T, loss_derivative)
            
            layer:Layers.LSTMMemoryCell
            current_layer_grad = np.dot(loss_derivative, self.Weights_hy.T)
            
            for layer, index in zip(reversed(self.Hidden_Layers), reversed(range(self.hidden_layer_count))):
                total_dh = current_layer_grad + dh_next[index]
                np.clip(total_dh, -1.0, 1.0, out=total_dh)
                if t > 0:
                    h_prev_t = hidden[t-1][index]
                else:
                    h_prev_t = np.zeros((1, self.hidden_size))

                if index == 0:
                    layer_input = inputs[t].reshape(1, -1)
                    split_point = self.input_size
                else:
                    layer_input = hidden[t][index-1]
                    split_point = self.hidden_size

                stack = np.concatenate((layer_input, h_prev_t), axis=1)
                d_input_gate_weights, d_input_gate_biases, d_input_weights, d_input_biases, \
                d_output_gate_weights, d_output_gate_biases, d_input_activation, d_state_cell = \
                        layer.cell_backward(stack, caches[t][index],total_dh,state_cell_next[index], t)
                GA_hidden_hh[index]["input_weight_gradients"] += d_input_weights
                GA_hidden_hh[index]["input_gate_weight_gradients"] += d_input_gate_weights
                GA_hidden_hh[index]["output_gate_weight_gradients"] += d_output_gate_weights
                GA_bias_h[index]["input_bias_gradients"] += d_input_biases
                GA_bias_h[index]["input_gate_bias_gradients"] += d_input_gate_biases
                GA_bias_h[index]["output_gate_bias_gradients"] += d_output_gate_biases
                state_cell_next[index]= d_state_cell 

                current_layer_grad = d_input_activation[:, :split_point]
                dh_next[index]= d_input_activation[:, split_point:]
                # np.clip(state_cell_next[index], -1.0, 1.0, out=state_cell_next[index])
                np.clip(dh_next[index], -1.0, 1.0, out=dh_next[index])

        return GA_weights_hy, GA_bias_y, GA_hidden_hh, GA_bias_h
                


class Layers():
    class LSTMMemoryCell():
        def __init__(self, input_size, hidden_size, initialization = "xavier", activation = "tanh"):
            self.hidden_size = hidden_size
            stacked_size = input_size+hidden_size

            if activation == "tanh":
                self.activation = modified_tanh
                self.activation_derivative = tanh_derivative
            else : 
                self.activation = modified_sigmoid
                self.activation_derivative= modified_sigmoid_derivative

            #Weight Initilization
            if initialization == "xavier": initialize = np.sqrt(1.0/input_size)
            else :initialize = 0.001
            self.input_weights = np.random.randn(stacked_size,hidden_size) * initialize
            self.input_gate_weights = np.random.randn(stacked_size, hidden_size) * initialize
            self.output_gate_weights = np.random.randn(stacked_size, hidden_size) * initialize

            #Biases Initialization
            self.input_biases = np.zeros(hidden_size)
            self.input_gate_biases = np.zeros(hidden_size)
            self.output_gate_biases = np.zeros(hidden_size)

            #State_Cell_Initialization
            self.state_cell = np.zeros(hidden_size)
            self.state_cell_history = []
            
            pass
        
        def cell_forward(self, stacked_input:np.ndarray) -> np.ndarray:
            input_signal = self.activation(np.dot(stacked_input, self.input_weights) + self.input_biases)
            input_gate_activation = sigmoid(np.dot(stacked_input, self.input_gate_weights) +self.input_gate_biases)
        
            self.state_cell += input_signal * input_gate_activation
            self.state_cell_history.append(self.state_cell.copy())

            output_signal = self.activation(self.state_cell)
            output_gate_activation = sigmoid(np.dot(stacked_input, self.output_gate_weights) +self.output_gate_biases)

            output_activation = output_signal*output_gate_activation

            math_cache = {
                "input_signal" : input_signal,
                "input_gate_activation": input_gate_activation,
                "output_signal": output_signal,
                "output_gate_activation": output_gate_activation
            }

            return output_activation, math_cache
        
        def cell_backward(self, stacked_input, cache, dh_next, state_cell_next, time_step):
            input_signal = cache["input_signal"]
            input_gate_activation = cache["input_gate_activation"]
            output_signal = cache["output_signal"]
            output_gate_activation = cache["output_gate_activation"]
            state_cell_t = self.state_cell_history[time_step]

            d_output_gate = dh_next*sigmoid_derivative(output_gate_activation)*self.activation(state_cell_t)
            d_output_gate_weights = np.dot(stacked_input.T, d_output_gate)
            d_output_gate_biases = np.sum(d_output_gate, axis=0)

            d_state_cell = dh_next * output_gate_activation * self.activation_derivative(state_cell_t) + state_cell_next

            d_input_gate = d_state_cell*sigmoid_derivative(input_gate_activation)*input_signal 
            d_input_gate_weights = np.dot(stacked_input.T, d_input_gate)
            d_input_gate_biases = np.sum(d_input_gate, axis=0)

            d_input = d_state_cell*tanh_derivative(input_signal)*input_gate_activation
            d_input_weights = np.dot(stacked_input.T, d_input)
            d_input_biases = np.sum(d_input, axis=0)

            d_input_activation = np.dot(d_input_gate, self.input_gate_weights.T) + \
                     np.dot(d_input, self.input_weights.T) + \
                     np.dot(d_output_gate, self.output_gate_weights.T)


            return d_input_gate_weights, d_input_gate_biases, d_input_weights, d_input_biases, d_output_gate_weights, d_output_gate_biases, d_input_activation, d_state_cell
            
        def update_cell_weights_biases(self, GA_weights, GA_biases, learning_rate):
            self.input_weights -= learning_rate*GA_weights["input_weight_gradients"]
            self.input_biases -= learning_rate*GA_biases["input_bias_gradients"]

            self.input_gate_weights -= learning_rate*GA_weights["input_gate_weight_gradients"]
            self.input_gate_biases -= learning_rate*GA_biases["input_gate_bias_gradients"]

            self.output_gate_weights -= learning_rate*GA_weights["output_gate_weight_gradients"]
            self.output_gate_biases -= learning_rate*GA_biases["output_gate_bias_gradients"]

        def reset_state(self):
            self.state_cell = np.zeros((1, self.hidden_size)) 
            self.state_cell_history = []