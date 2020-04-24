import numpy as np
import time
import scipy
from scipy.signal import convolve2d

try:
    import _pickle as pickle
except ImportError:
    import cPickle as pickle


class CNN2D(object):
    def __init__(self, inp_size, out_shape, args, manual_mode=True):
        if manual_mode:
            self.args = args

            self.num_layers = len(self.args)
            self.num_conv_layers = 0
            self.num_dense_layers = 0

            self.conv_kernel = []
            self.conv_bias = []
            self.dense_weights = []
            self.dense_bias = []

            channels = inp_size[0]
            output_size = inp_size
            for arg in args:
                if arg["type"] == "conv2d":
                    kernels = arg["kernel"]
                    num_kernels = kernels[0]
                    kernels_shape = (num_kernels,) + (channels,) + kernels[1:]
                
                    self.conv_kernel.append(np.ones(kernels_shape))
                    self.conv_bias.append(np.zeros((num_kernels,)))

                    channels = num_kernels
                    output_size = self._calculate_output_size(kernels_shape, arg["padding"], arg["stride"], output_size)

                    self.num_conv_layers += 1
            
                elif arg["type"] == "dense":
                    self.dense_weights.append(np.zeros(shape=(np.prod(output_size), arg["size"])))  
                    self.dense_bias.append(np.zeros(shape=(1, arg["size"])))

                    self.num_dense_layers += 1
                    output_size = arg["size"]

            self.weights = self.conv_kernel + self.conv_bias + self.dense_weights + self.dense_bias
        else:
            pass

    def _output_size(self, l_input, l_kernel, l_pad, stride):
        return ((l_input - l_kernel + 2*l_pad) / stride) + 1

    def _pad_input(self, inp, layer_padding):
        padding_h = layer_padding[0]
        padding_w = layer_padding[1]

        if padding_h + padding_w == 0:
            return inp
        
        padded = inp
        shape = padded.shape

        if padding_h > 0:
            h_padding_shape = (shape[0], padding_h, shape[2])
            padded = np.concatenate((np.zeros(h_padding_shape), padded, np.zeros(h_padding_shape)), axis=1)
            shape = padded.shape

        if padding_w > 0:
            w_padding_shape = (shape[0], shape[1], padding_w)
            padded = np.concatenate((np.zeros(w_padding_shape), padded, np.zeros(w_padding_shape)), axis=2) 
            shape = padded.shape

        return padded

    def _activation_layer(self, output_layer, activation_name, activation_args):
        if activation_name is None:
            return output_layer

        if activation_name == "sigmoid":
            return 1 / (1 + np.exp(-output_layer))

        if activation_name == "relu":
            return np.maximum(0, output_layer)

        if activation_name == "tanh":
            return np.tanh(output_layer)

        if activation_name == "softmax":
            exp = np.exp(output_layer)
            sum_exp = np.sum(exp)
            return exp / sum_exp

        if activation_name == "column-softmax":
            channels = []
            for c in range(output_layer.shape[0]):
                exp = np.exp(output_layer[c])
                normalized = exp / exp.sum(0)
                channels.append(np.expand_dims(normalized, axis=0))
            return np.concatenate(channels, axis=0)

        if activation_name == "cross-entropy":
            exp = np.exp(output_layer)
            sum_exp = np.sum(exp)
            return -np.log(exp / sum_exp)

    def _calculate_output_size(self, kernels_shape, layer_padding, layer_stride, inp_shape):

        num_kernels = kernels_shape[0]
        kernel_d = kernels_shape[1]
        kernel_h = kernels_shape[2]
        kernel_w = kernels_shape[3]

        padding_h = layer_padding[0]
        padding_w = layer_padding[1]

        stride_h = layer_stride[0]
        stride_w = layer_stride[1]

        inp_d = inp_shape[0]
        inp_h = inp_shape[1]
        inp_w = inp_shape[2]

        output_dim = (num_kernels,
                      int(self._output_size(inp_h, kernel_h, padding_h, stride_h)),
                      int(self._output_size(inp_w, kernel_w, padding_w, stride_w)))

        return output_dim

    # TODO: fix for stride > 1. 
    def _2D_Conv_fast(self, inp, output_dim, layer_kernel_weights, layer_bias_weights, layer_stride, kernels_shape):

        stride_h = layer_stride[0]
        stride_w = layer_stride[1]

        num_kernels = kernels_shape[0]
        kernel_d = kernels_shape[1]
        kernel_h = kernels_shape[2]
        kernel_w = kernels_shape[3]

        final_result = []
        for d in range(num_kernels):
            kernel = layer_kernel_weights[d]
            bias = layer_bias_weights[d]

            kernel_result = None
            for c in range(kernel_d):
                kernel_channel_result = convolve2d(inp[c], kernel[c], mode='valid')
                if kernel_result is None:
                    kernel_result = kernel_channel_result
                else:
                    kernel_result = kernel_result + kernel_channel_result

            kernel_result = kernel_result + bias

            final_result.append(np.expand_dims(kernel_result, axis=0))

        return np.concatenate(final_result, axis=0) 

    def _2D_Conv_iterative(self, inp, output_dim, layer_kernel_weights, layer_bias_weights, layer_stride, kernels_shape):
        output_layer = np.zeros(output_dim)

        stride_h = layer_stride[0]
        stride_w = layer_stride[1]

        num_kernels = kernels_shape[0]
        kernel_d = kernels_shape[1]
        kernel_h = kernels_shape[2]
        kernel_w = kernels_shape[3]

        for d in range(output_dim[0]):
            current_kernel = layer_kernel_weights[d]
            current_bias = layer_bias_weights[d]
            for h in range(output_dim[1]):
                y = h*stride_h
                inp_slice = inp[:, y:y+kernel_h, :]
                for w in range(output_dim[2]):
                    x = w*stride_w
                    output_layer[d, h, w] = np.sum(current_kernel * inp_slice[:, :, x:x+kernel_w]) + current_bias

        return output_layer

    # Down-samples layer, depth remains unchanged of layer. 
    def _Maxpool(self, inp, output_dim, layer_kernel_weights, layer_bias_weights, layer_stride, kernels_shape):
        pass

    def _Dense(self, inp, layer_dense_weights, layer_dense_bias):
        out = np.expand_dims(inp.flatten(), 0)
        out = np.dot(out, layer_dense_weights) + layer_dense_bias
        return out

    def predict(self, inp):
        inference_t = 0.0
        i_conv = 0
        i_dense = 0
        for layer in range(self.num_layers):
            arg = self.args[layer]

            layer_activation_name, layer_activation_args = arg["activation"] 

            layer_type = arg["type"]

            # TODO: inp_shape is replicated from self.inp_shape

            start = time.time()
            if layer_type == "conv2d":
                layer_kernel_weights = self.conv_kernel[i_conv]
                layer_bias_weights = self.conv_bias[i_conv]

                kernels_shape = layer_kernel_weights.shape
                layer_padding = arg["padding"]
                layer_stride = arg["stride"]

                inp_shape = inp.shape

                inp = self._pad_input(inp, layer_padding)

                output_dim = self._calculate_output_size(kernels_shape,
                                                         layer_padding,
                                                         layer_stride,
                                                         inp_shape)

                output_layer = self._2D_Conv_fast(inp, 
                                                  output_dim, 
                                                  layer_kernel_weights, 
                                                  layer_bias_weights, 
                                                  layer_stride,
                                                  kernels_shape)
                i_conv += 1
            elif layer_type == "maxpool":
                output_layer = self._Maxpool(inp,
                                             output_dim,
                                             layer_kernel_weights,
                                             layer_bias_weights,
                                             layer_stride,
                                             kernels_shape)

            elif layer_type == "dense":
                layer_dense_weights = self.dense_weights[i_dense]
                layer_dense_bias = self.dense_bias[i_dense]
                
                output_layer = self._Dense(inp,
                                           layer_dense_weights, 
                                           layer_dense_bias)
                i_dense += 1


            inference_t += time.time() - start  

            # TODO: Add assert statements to check sizes, etc...

            # TODO: Move all of this code to a separate Conv function
            # TODO: Differentiate between conv and maxpool layers 

            inp = self._activation_layer(output_layer, 
                                         activation_name=layer_activation_name, 
                                         activation_args=layer_activation_args)

        out = inp

        #print("Inference time: %0.3f" % (inference_t))

        return out

    def get_weights(self):
        return self.weights

    def get_parameter_count(self):
        total = 0
        for w in self.weights:
            total += np.prod(w.shape)

        return total 

    def set_weights(self, weights):
        self.weights = weights
        i = 0

        self.conv_kernel = self.weights[i:self.num_conv_layers]
        i = self.num_conv_layers

        self.conv_bias = self.weights[i:i+self.num_conv_layers]
        i = 2*self.num_conv_layers

        self.dense_weights = self.weights[i:i+self.num_dense_layers]
        i = i + self.num_dense_layers

        self.dense_bias = self.weights[i:i+self.num_dense_layers]

    def save(self, filename='weights.pkl'):
        with open(filename, 'wb') as fp:
            pickle.dump(self.weights, fp)

    def load(self, filename='weights.pkl'):
        with open(filename, 'rb') as fp:
            self.weights = pickle.load(fp)
