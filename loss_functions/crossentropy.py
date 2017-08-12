from __future__ import division
import numpy as np

from lossfunction import LossFunction

class CrossEntropy(LossFunction):

    def compute(self, observed, target):
        return -(target * np.log(observed)).sum()

    def error(self, observed,target):
        return -(target / observed)


