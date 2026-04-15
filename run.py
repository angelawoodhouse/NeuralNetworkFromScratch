# Entry point
# Import packages and dependencies
from src.data_loader import load_and_split_data
from src.train import gradient_descent, test_prediction


def main():
    # Load data
    X_train, Y_train, X_dev, Y_dev = load_and_split_data()

    # Train
    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 500, 0.10)

    # Results
    print('\nTesting predictions on training set: ')
    for i in range(4):
        test_prediction(i, X_train, Y_train,  W1, b1, W2, b2)

if __name__ == "__main__":
    main()