from __future__ import division
import math
import numpy as np
import sys

sys.path.append('./')
from activations.sigmoid import Sigmoid
from activations.relu import ReLU
from optimizers.adagrad_optimizer import AdagradOptimizer
from initializers.initialization_functions import uniform


class Layer:
    def __init__(self, layer_size, input_sizes, init_function=uniform, activation=Sigmoid, alpha=0.01,
                 optimizer=AdagradOptimizer):
        self.alpha = alpha
        self.layer_size = layer_size
        self.input_sizes = input_sizes
        self.activation = activation()

        # Setup weights
        self.weights = [init_function(layer_size, input_size)
                        for input_size in input_sizes]
        self.weight_optimizers = [optimizer() for input_size in input_sizes]
        self.deltas = [np.zeros((layer_size, input_size)) for input_size in input_sizes]

        if activation == ReLU:
            self.bias = np.ones((layer_size, 1))
        else:
            self.bias = init_function(layer_size, 1)
        # Setup bias
        self.bias_optimizer = optimizer()
        self.del_bias = np.zeros((layer_size, 1))
        # Setup input and activation stacks for BackProp
        self.input_stack = []
        self.activation_stack = []

    def print_w(self):
        print "Weight : ", self.weights
        print "Bias : ", self.bias

    def forward(self, input_list, train=False):
        temp = self.bias
        for index, input_ in enumerate(input_list):
            if type(input_) == type([]):
                input_ = np.array(input_)
            reshaped = input_.reshape((len(input_), 1))
            input_list[index] = reshaped
            temp = temp + self.weights[index].dot(reshaped)
        act = self.activation.apply(temp)
        if train:
            self.activation_stack.append(act)
            self.input_stack.append(input_list)
        return act

    def backward(self, error_incoming, apply=True, alpha=0.01):
        error_incoming = np.array(error_incoming).reshape((self.layer_size, 1)) if type(error_incoming) == type(
            []) else error_incoming
        error_lower = error_incoming * self.activation.differentiate(self.activation_stack.pop())
        inputs = self.input_stack.pop()
        self.compute_gradients(error_lower, inputs)
        error_outgoing = [weight.T.dot(error_lower) for weight in self.weights]
        if apply:
            self.apply_gradients(alpha)
        return error_outgoing

    def compute_gradients(self, error_lower, inputs):
        for index, input_ in enumerate(inputs):
            self.deltas[index] = self.deltas[index] + error_lower.dot(input_.T)
        self.del_bias = self.del_bias + error_lower

    def apply_gradients(self, alpha=0.01):
        for index, delta in enumerate(self.deltas):
            self.weights[index] = self.weight_optimizers[index].update(self.weights[index], delta, alpha)
        self.bias = self.bias_optimizer.update(self.bias, self.del_bias, alpha)

        self.deltas = [np.zeros((self.layer_size, input_size)) for input_size in self.input_sizes]
        self.del_bias = np.zeros((self.layer_size, 1))


