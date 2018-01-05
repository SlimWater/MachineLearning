import numpy as np


class ConvLayer(object):
    def __init__(self, input_width, input_height, channel_number, filter_width, filter_height, filter_number, zero_padding, stride, activator, learning_rate):
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.filter_number = filter_number
        self.zero_padding = zero_padding
        self.stride = stride
        self.filters = []
        # W = (W_input - filter_width + 2* zero_padding)/stride +1
        # H = (H_input - filter_width + 2*zero_padding)/stride +1
        self.output_width = (self.input_width - self.filter_width + 2*self.zero_padding)/self.stride +1
        self.output_height= (self.input_height - self.filter_height + 2*self.zero_padding)/self.stride +1
        self.output_array = np.zeros((int(self.filter_number), int(self.output_height), int(self.output_width)))
        for i in range(filter_number):
            self.filters.append(Filter(filter_width, filter_height, channel_number))
        self.activator = activator
        self.learning_rate = learning_rate

    def forward(self, input_array):
        self.input_array = input_array
        self.padded_input_array = self.padding(input_array, self.zero_padding)
        for f in range(self.filter_number):
            filter = self.filters[f]
            ConvLayer.conv(self.padded_input_array, filter.get_weights(), self.output_array[f], self.stride, filter.get_bias())
        #element_wist_op(self.output_array, self.activator.forward) activation function implementation

    def padding(self, input_array, zp):
        if zp == 0:
            return input_array
        else:
            if input_array.ndim == 3:
                input_width = input_array.shape[2]
                input_height = input_array.shape[1]
                input_depth = input_array.shape[0]
                padded_array = np.zeros((input_depth, input_height+2*zp, input_width+2*zp))
                padded_array[:,zp:zp+input_height, zp:zp+input_width] = input_array

            elif input_array.ndim == 2:
                input_width = input_array.shape[1]
                input_height = input_array.shape[0]
                padded_array = np.zeros((input_height+2*zp, input_width + 2*zp))
                padded_array[zp:zp+input_height, zp:zp+input_width] = input_array
            else:
                padded_array = input_array
                print("\r\n Input array dimenstion is not supported")
            return padded_array

    def conv(self, input_array, kernel_array, output_array, stride, bias):
        #the output_array is a 3d matrix in this module. in this function the input is output_array[f]
        output_width = output_array.shape[1]
        output_height = output_array.shape[0]
        kernel_width = kernel_array.shape[-1]
        kernel_height = kernel_array.shape[-2]
        if input_array.ndim == 2:
            for i in range(output_height):
                for j in range(output_width):
                    #numpy array range array[a:b] means a->b-1
                    temp_array = input_array[i*stride:i*stride+kernel_height, j*stride:j*stride+kernel_width]
                    output_array[i,j] = np.multiply(temp_array, kernel_array).sum() + bias

        elif input_array.ndim == 3:
            for i in range(output_height):
                for j in range(output_width):
                    for m in range(input_array.shape[0]):
                        if m == 0:
                            temp_array_input = input_array[m, i*stride:i*stride+kernel_height, j*stride:j*stride+kernel_width]
                            temp_array_filter = kernel_array
                        else:
                            temp_array_input = np.concatenate((temp_array_input, input_array[m, i*stride:i*stride+kernel_height, j*stride:j*stride+kernel_width]), axis = 0)
                            temp_array_filter = np.concatenate((temp_array_filter, kernel_array), axis = 0)
                    output_array[i, j] = np.multiply(temp_array_input, temp_array_filter).sum() + bias


    #now implement the training functions
    def bp_sensitivity_map(self, sensitivity_array, activator):
        #sensitivity_array: next layer sensitivity_map
        #next layer activation function
        #to calculate this layer sensitivity_map
        expanded_array = self.expand_sensitivity_map(sensitivity_array)
        # zp is always 1
        zp = (self.input_width + self.filter_width - 1 - expanded_array.shape[2])/2
        padded_expanded_array = self.padding(expanded_array, zp)



    def expand_sensitivity_map(self, sensitivity_array):
        # convert stride S array into stride 1 array
        depth = sensitivity_array.shape[0]
        expanded_width = self.input_width - self.filter_width + 2*self.zero_padding +1
        expanded_height = self.input_height - self.filter_height + 2 * self.zero_padding + 1
        expand_array =  np.zeros((depth, expanded_height,expanded_width))
        for i in range(self.output_height):
            for j in range(self.output_width):
                i_pos = i*self.stride
                j_pos = j*self.stride
                expand_array[:,i_pos,j_pos] = sensitivity_array[:,i, j]
        return expand_array

class Filter(object):
    def __init__(self, width, height, depth):
        self.weights = np.random.uniform(-1e-4, 1e-4,(depth,height, width))
        self.bias = 0
        self.weights_grad = np.zeros(self.weights.shape)
        self.bias_grad = 0
    def __repr__(self):
        return "filter weights:\n%s\nbias:\n%s" (repr(self.weights), repr(self.bias))
    def get_wieghts(self):
        return self.weights
    def get_bias(self):
        return self.bias
    def update(self, learning_rate):
        self.weights = self.weights-learning_rate*self.weights_grad
        self.bias = self.bias - learning_rate*self.bias_grad

