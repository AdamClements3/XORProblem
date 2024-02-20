from imports import *
def _sigmoid(self, x):
    """
    The sigmoid activation function.
    """
    return 1 / (1 + np.exp(-x))

def _delsigmoid(self, x):
    """
    The first derivative of the sigmoid function wrt x
    """
    return x * (1 - x)