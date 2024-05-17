from tqdm import tqdm
from typing import List

from nn.layers import *


class NeuralNetwork:
    def __init__(self, layers: List[Layer]):
        self.layers = layers
        self.loss = None
        self.train_history = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forwards pass through the whole network.

        :param x: Input to the network.
        :return: Output of the network.
        """

        for layer in self.layers:
            x = layer.forward(x)

        return x

    def backward(self, x: np.ndarray, y: np.ndarray, lr: float) -> None:
        """
        Backward pass through the whole network.

        :param x: Input to the network.
        :param y: Ground truth output.
        :param lr: Learning rate.
        :return: None
        """
        output = self.forward(x)

        mse = MSE()
        self.loss = mse.forward(output, y)
        op_grad = mse.backward(output, y)

        for layer in reversed(self.layers):
            op_grad = layer.backward(op_grad)
            layer.update_params(lr)

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int = 100, lr: float = 0.01) -> None:
        """
        Train the neural network.

        :param x: Input to the network.
        :param y: Ground truth output.
        :param epochs: Number of epochs for training.
        :param lr: Learning rate.
        :return: None
        """

        # enforce x and y to be 2D arrays
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        if len(y.shape) == 1:
            y = y.reshape(1, -1)

        pbar = tqdm(range(epochs))

        for _ in pbar:
            epoch_loss = 0
            for i in range(x.shape[0]):
                self.backward(x[i:i+1], y[i:i+1], lr)
                epoch_loss += self.loss

            epoch_loss /= x.shape[0]
            pbar.set_description(f"Loss: {epoch_loss:.5f}")
            self.train_history.append(epoch_loss)
