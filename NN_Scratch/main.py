import numpy as np
import matplotlib.pyplot as plt
from nn.layers import Linear, LeakyReLU
from nn.network import NeuralNetwork


x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork([
    Linear(2, 8),
    LeakyReLU(),
    Linear(8, 1),
    LeakyReLU(),
])

nn.train(x, y, epochs=1000, lr=0.01)

# Test the neural network
print(nn.forward(x))

plt.plot(nn.train_history)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
