from __future__ import division
import numpy as np
from activation import Activation

class Sigmoid(Activation):

    def apply(self, input_):
        return 1 / (1 + np.exp(-1 * input_))

    def differentiate(self, activation):
        return activation * (1 - activation)
