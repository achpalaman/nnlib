from __future__ import division
import numpy as np

from activation import Activation

class Tanh(Activation):
    def apply(self, input_):
        return np.tanh(input_)

    def differentiate(self, activation):
        return (1 - activation * activation)
