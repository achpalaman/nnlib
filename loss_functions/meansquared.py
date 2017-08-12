from __future__ import division
import numpy as np

from lossfunction import LossFunction

class MeanSquared(LossFunction):

    def compute(self, observed, target):
        #assert type(observed) == type(target) == 'numpy.ndarray'
        return 0.5 * ((observed - target) ** 2).sum()

    def error(self, observed, target):
        #assert type(observed) == type(target) == 'numpy.ndarray'
        return (target - observed)

