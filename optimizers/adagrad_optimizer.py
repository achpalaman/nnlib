from optimizer import Optimizer
import numpy as np

class AdagradOptimizer(Optimizer):
    def update(self, weight, delta, alpha):
        if self.mem is None:
            # initialize
            self.mem = np.zeros_like(weight)
        self.mem += delta * delta
        # adagrad update rule
        return weight + alpha * delta.clip(-5, 5) / np.sqrt(self.mem + 1e-8)

    def __init__(self):
        super(AdagradOptimizer, self).__init__()
        # Not initialized yet as we don't know the size
        # Will be initialized when update is called for the first time
        self.mem = None
