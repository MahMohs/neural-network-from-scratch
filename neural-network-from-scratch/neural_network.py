import numpy as np


class NeuralNetwork:

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.learning_rate = learning_rate

        # Initialize weights and biases
        self.W1 = np.abs(np.random.randn(input_size, hidden_size)) * 0.1
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def forward(self, X):
        # First layer
        self.Z1 = X @ self.W1 + self.b1
        self.H = self.relu(self.Z1)

        # Output layer
        self.Z2 = self.H @ self.W2 + self.b2
        self.y_hat = self.Z2

        return self.y_hat

    def loss(self, y_hat, y):
        return np.mean((y_hat - y) ** 2) / 2

    def backward(self, X, y):
        N = X.shape[0]

        # Gradient of loss with respect to prediction
        dL_dy = (self.y_hat - y) / N

        # Output layer gradients
        dW2 = self.H.T @ dL_dy
        db2 = np.sum(dL_dy, axis=0, keepdims=True)

        # Gradient flowing into hidden layer
        dH = dL_dy @ self.W2.T

        # Through ReLU
        dZ1 = dH * self.relu_derivative(self.Z1)

        # First layer gradients
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Update parameters
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):

            y_hat = self.forward(X)

            current_loss = self.loss(y_hat, y)

            self.backward(X, y)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {current_loss:.6f}")
