from __future__ import division
import numpy as np

from activation import Activation

class ReLU(Activation):
    def apply(self, input_):
        return np.maximum(input_, 0)

    def differentiate(self, activation):
        return np.ones(max(activation.shape)).reshape(activation.shape) \
                                         * (activation > 0)
