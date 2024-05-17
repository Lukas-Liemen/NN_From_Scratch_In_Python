import numpy as np


class Layer:
    def __init__(self):
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of the layer.

        :param x: Input to the activation function.
        :return: Output of the activation function.
        """
        raise NotImplementedError

    def backward(self, op_grad: np.ndarray) -> np.ndarray:
        """
        Backward pass of the layer.

        :param op_grad: Gradient of the loss from the next layer.
        :return: Gradient of the loss with respect to the input.
        """
        raise NotImplementedError

    def update_params(self, lr: float = 0.01) -> None:
        """
        Update the parameters of the layer.

        :param lr: Learning rate.
        :return: None
        """
        raise NotImplementedError


class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return np.maximum(0, x)

    def backward(self, op_grad: np.ndarray) -> np.ndarray:
        return op_grad * (self.input > 0)

    def update_params(self, lr: float = 0.01) -> None:
        """
        No parameters to update for activation functions.
        """
        pass


class LeakyReLU(Layer):
    def __init__(self, alpha: float = 0.001):
        super().__init__()
        self.alpha = alpha
        self.input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, op_grad: np.ndarray) -> np.ndarray:
        return op_grad * np.where(self.input > 0, 1, self.alpha)

    def update_params(self, lr: float = 0.01) -> None:
        """
        No parameters to update for activation functions.
        """
        pass


class Linear(Layer):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.input = None
        self.weights = np.random.randn(in_features, out_features)
        self.bias = np.random.randn(1, out_features)
        self.d_weights = np.zeros_like(self.weights)
        self.d_bias = np.zeros_like(self.bias)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return np.dot(x, self.weights) + self.bias

    def backward(self, op_grad: np.ndarray) -> np.ndarray:
        if len(op_grad.shape) == 1:
            op_grad = op_grad.reshape(1, -1)

        self.d_weights = np.dot(self.input.T, op_grad)
        self.d_bias = np.sum(op_grad, axis=0, keepdims=True)

        return np.dot(op_grad, self.weights.T)

    def update_params(self, lr: float = 0.01) -> None:
        self.weights -= lr * self.d_weights
        self.bias -= lr * self.d_bias


class MSE:
    def __init__(self):
        pass

    @staticmethod
    def forward(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.mean((x - y) ** 2)

    @staticmethod
    def backward(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return -2 * (y - x) / x.shape[0]
