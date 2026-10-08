import numpy as np

from neural_network import NeuralNetwork


# Training data
X = np.array([
    [1.0, 2.0],
    [2.0, 3.0],
    [3.0, 4.0],
    [4.0, 5.0],
    [5.0, 6.0],
    [6.0, 7.0]
])

y = np.array([
    [5.0],
    [8.0],
    [11.0],
    [14.0],
    [17.0],
    [20.0]
])


# Create model
model = NeuralNetwork(
    input_size=2,
    hidden_size=4,
    output_size=1,
    learning_rate=0.01
)


# Train
model.train(X, y, epochs=1000)


# Final predictions
predictions = model.forward(X)

print("\nPredictions:")
print(predictions)

print("\nActual:")
print(y)