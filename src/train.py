# Training loop and orchestration
# Import packages and dependencies
import matplotlib.pyplot as plt
from src.model import init_params, forward_prop, back_prop, update_params, make_predictions
from src.utils import get_predictions, get_accuracy

## Train the neural network using gradient descent
def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params() # Initialise parameters randomly
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X) # Forward propagation (make predictions)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y) # Backpropagation (compute gradients)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha) # Update parameters
        if i % 10 == 0: # Print progress every 10 iterations
            print('Iteration: ', i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2

# Test predictions for results
def test_prediction(index, X_data, Y_data, W1, b1, W2, b2):
    current_image = X_data[:, index, None]
    prediction = make_predictions(X_data[:, index, None], W1, b1, W2, b2)
    label = Y_data[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()