import numpy as np
import math


def xavier(a, b):
    return np.random.randn(a, b) * 2.0 / (math.sqrt(a) + math.sqrt(b))


def uniform(a, b):
    return np.random.uniform(low=-0.01, high=0.01, size=(a, b))


def random(a, b):
    return np.random.randn(a, b)
