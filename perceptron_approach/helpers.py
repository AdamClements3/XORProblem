from imports import *

def _gradient(self, node, exp, output):
    """
    Return the gradient for a weight.
    This is the value of delta-w.
    """
    return node * (exp - output)

def update_weights(self, exp, output):
    """
    Update weights and bias based on their respective gradients
    """
    for i in range(self.input_nodes):
        self.w[i] += self.lr * self._gradient(self.node_val[i], exp, output)

    # the value of the bias node can be considered as being 1 and the weight between this node
    # and the output node being self.b
    self.b += self.lr * self._gradient(1, exp, output)

def forward(self, datapoint):
    """
    One forward pass through the perceptron.
    Implementation of "wX + b".
    """
    return self.b + np.dot(self.w, datapoint)

def classify(self, datapoint):
    """
    Return the class to which a datapoint belongs based on
    the perceptron's output for that point.
    """
    if self.forward(datapoint) >= 0:
        return 1