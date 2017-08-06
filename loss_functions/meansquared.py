from __future__ import division
import numpy as np

from lossfunction import LossFunction

class MeanSquared(LossFunction):

    def compute(observed, target):
        assert type(observed) == type(target) == 'numpy.ndarray'
        return 0.5 * ((observed - target) ** 2).sum()

    def error(observed, target):
        assert type(observed) == type(target) == 'numpy.ndarray'
        return (target - observed)

