from abc import ABCMeta, abstractmethod

class Activation:
    __metaclass__ = ABCMeta

    @abstractmethod
    def apply(self, input_):
        """ Apply the activation function on input_.

        Keyword arguments:
        input_ -- the input sample on which the
                  activation function is to be
                  applied
        """
        pass

    @abstractmethod
    def differentiate(self, activation, target=None):
        """ Compute the local gradient.

        Keyword arguments:
        activation -- the activation value of
                      the layer for which the
                      local gradient is to be
                      calculated
        """
        pass
