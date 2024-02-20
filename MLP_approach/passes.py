from imports import *

def forward(self, batch):
    """
    A single forward pass through the network.
    Implementation of wX + b
    """

    self.hidden_ = np.dot(batch, self.weights_01) + self.b01
    self.hidden_out = self._sigmoid(self.hidden_)

    self.output_ = np.dot(self.hidden_out, self.weights_12) + self.b12
    self.output_final = self._sigmoid(self.output_)

    return self.output_final

def update_weights(self):

    # Calculate the squared error
    loss = 0.5 * (self.target - self.output_final) ** 2
    print(loss)
    self.losses.append(np.sum(loss))

    error_term = (self.target - self.output_final)

    # the gradient for the hidden layer weights
    grad01 = self.train_data.T @ (((error_term * self._delsigmoid(self.output_final)) * self.weights_12.T) * self._delsigmoid(self.hidden_out))
    print("grad01: ", grad01)
    print(grad01.shape)

    # the gradient for the output layer weights
    grad12 = self.hidden_out.T @ (error_term * self._delsigmoid(self.output_final))

    print("grad12: ", grad12)
    print(grad12.shape)

    # updating the weights by the learning rate times their gradient
    self.weights_01 += self.lr * grad01
    self.weights_12 += self.lr * grad12

    # update the biases the same way
    self.b01 += np.sum(self.lr * ((error_term * self._delsigmoid(self.output_final)) * self.weights_12.T) * self._delsigmoid(self.hidden_out), axis=0)
    self.b12 += np.sum(self.lr * error_term * self._delsigmoid(self.output_final), axis=0)
    
def classify(self, datapoint):
    """
    Return the class to which a datapoint belongs based on
    the perceptron's output for that point.
    """
    datapoint = np.transpose(datapoint)
    if self.forward(datapoint) >= 0.5:
        return 1

    return 0