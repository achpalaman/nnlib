from optimizer import Optimizer


class GdOptimizer(Optimizer):
    def update(self, weight, delta, alpha):
        return weight + alpha * delta
