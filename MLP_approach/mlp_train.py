from imports import *
def train(self):
    """
    Train an MLP. Runs through the data num_epochs number of times.
    A forward pass is done first, followed by a backward pass (backpropagation)
    where the networks parameter's are updated.
    """
    for _ in range(self.num_epochs):
        self.forward(self.train_data)
        self.update_weights()