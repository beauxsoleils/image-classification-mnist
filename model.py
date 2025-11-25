#!/usr/bin/env python3

from torch.nn import Module, Linear, ReLU, Sequential, Flatten, BatchNorm1d, Dropout

class Model(Module):
    """
    Class Model defines a forward-feed neural network with an input layer, three hidden layers and a single output layer.
    Here we use it to perform image classification on the 10-digit MNIST dataset.
    """

    def __init__(self):
        super().__init__()
        self.net = Sequential(
            Flatten(),

            Linear(784, 512), # input layer (784 features) to hidden layer (512 neurons)
            BatchNorm1d(512), # normalizes each batch’s activations per feature to have zero mean and unit variance. 
            ReLU(), # activation function: rectified linear unit
            Dropout(0.1), # randomly zeroes a fraction p of activations during training.

            Linear(512, 128), # hidden layer
            BatchNorm1d(128),#batch normalization
            ReLU(), # activation function
            Dropout(0.1),

            Linear(128, 10) # hidden layer to output layer (10 classes for digits 0-9)
        )

    def forward(self, x):
        return self.net(x)

if __name__ == '__main__': pass




