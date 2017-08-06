from abc import ABCMeta, abstractmethod

class LossFunction:
    __metaclass__ = ABCMeta

    @abstractmethod
    def compute(self, observed, target):
        """ Compute the loss for the given sample.

        Keyword arguments:
        observed -- The observed activations of the network
        target   -- The expected activations of the network
        """
        pass

    @abstractmethod
    def error(self, observed, target):
        """ Compute the error for the sample.

        Keyword arguments:
        observed -- The observed activations of the network
        target   -- The expected activations of the network
        """
        pass
