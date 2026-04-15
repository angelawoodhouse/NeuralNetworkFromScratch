# Neural network architecture and training maths
# Import dependencies
import numpy as np
from src.utils import ReLU, derivReLU, softmax, one_hot, get_predictions

# Create the network structure with random starting values
def init_params():
    W1 = np.random.rand(10, 784) - 0.5 # Matrix of weights connecting the input layer to the first hidden layer
    b1 = np.random.rand(10, 1) - 0.5 # A bias for ach of the 10 hidden neurons
    W2 = np.random.rand(10, 10) - 0.5 # Weights connecting the first hidden layer to the output layer
    b2 = np.random.rand(10, 1) - 0.5 # Bias for each of the 10 output neurons
    return W1, b1, W2, b2

# Pass X through the network to compute predictions
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1 # Hidden layer: linear transformation
    A1 = ReLU(Z1) # Hidden layer: non-linear activation
    Z2 = W2.dot(A1) + b2 # Output layer: linear transformation
    A2 = softmax(Z2) # Output layer: convert to probabilities
    return Z1, A1, Z2, A2

# Compute gradients for all weights and biases
def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * derivReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

# Update parameters
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

# Predictions for results
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions