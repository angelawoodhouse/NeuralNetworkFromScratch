# Contains helper functions that do not depend on any other files
# Import packages
import numpy as np
import matplotlib.pyplot as plt

# Keep positive values, zero out negative values - allows the network to learn non-linear patterns
def ReLU(Z):
    return np.maximum(Z, 0)

def derivReLU(Z):
    return Z > 0

# Convert raw scores into probabilities
def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

# Convert integer labels into one-hot encoded vectors
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1)) # Create a matrix of zeros
    one_hot_Y[np.arange(Y.size), Y] = 1 # For each example, set the appropriate class column to 1
    one_hot_Y = one_hot_Y.T # Transpose so classes are in rows, examples are in columns
    return one_hot_Y

# Convert probability matrix to class predictions
def get_predictions(A2):
    return np.argmax(A2, 0)

# Calculate classification accuracy
def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

