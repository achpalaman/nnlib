from abc import ABCMeta, abstractmethod

class Optimizer:
    __metaclass__ = ABCMeta

    @abstractmethod
    def update(self, weight, delta, alpha):
        """ Update the weight with the delta.

        Keyword arguments:
        weight -- the weight matrix to be updated

        delta -- the error computed for that weight matrix
        """
        pass