
from torch.nn import Module, Sequential, Conv2d, ReLU, MaxPool2d, Dropout, Linear, Flatten


class ConvNeuralNet(Module):
    """
    """

    def __init__(self):
        super().__init__()

        # convolutional layer
        self.features = Sequential(
            Conv2d(1, 32, kernel_size=3, padding=1),  # 32 x 28 x 28
            ReLU(),
            Conv2d(32, 64, kernel_size=3, padding=1), # 64 x 28 x 28
            ReLU(),
            MaxPool2d(2, 2),                          # 64 x 14 x 14
            Dropout(0.25),
        )

        # classifier layer
        self.classifier = Sequential(
            Flatten(),                                # 64*14*14
            Linear(64 * 14 * 14, 128),
            ReLU(),
            Dropout(0.25),
            Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

if __name__ == '__main__': pass