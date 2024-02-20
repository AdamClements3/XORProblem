from imports import *

def train(self):
    """
    Train a single layer perceptron.
    """
    # the number of consecutive correct classifications
    correct_counter = 0

    for train, target in cycle(zip(self.train_data, self.target)):
        # end if all points are correctly classified
        if correct_counter == len(self.train_data):
            break

        output = self.classify(train)
        self.node_val = train

        if output == target:
            correct_counter += 1
        else:
            # if incorrectly classified, update weights and reset correct_counter
            self.update_weights(target, output)
            correct_counter = 0