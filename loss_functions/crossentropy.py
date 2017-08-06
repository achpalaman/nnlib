from __future__ import division
import numpy as np

from lossfunction import LossFunction

class CrossEntropy(LossFunction):

    def compute(observed, target):
        return -(target * np.log(observed)).sum()

    def error(observed,target):
        return -(target / observed)


